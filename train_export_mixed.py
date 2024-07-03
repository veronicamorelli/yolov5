import os
import sys
from pathlib import Path
import argparse

# Set the path to the YOLOv5 repository
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_train_export_args():

    parser = argparse.ArgumentParser(description='Train')

    # Training arguments
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default="clearml://63101d33dd984dddac2a0b855950a6cc", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=1, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="Personal_Experiments/veronica", help="save to project/name")
    parser.add_argument("--name", default="yolov5_unique_temp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    # Export arguments
    parser.add_argument("--keras", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--per-tensor", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workspace", type=int, default=4)
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--topk-per-class", type=int, default=100)
    parser.add_argument("--topk-all", type=int, default=100)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument('--class-map', type=list, nargs="+", default=["cart"])
    parser.add_argument(
        "--include",
        nargs="+",
        default=["onnx"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )


    return parser.parse_args()
    

# Train the model
def train_model(train_args):
    # Import train function from train.py
    from train import main as train_main

    opt = train_args
    # Run the training
    model_path = train_main(opt)
    
    return model_path, opt.imgsz


# Export the model
def export_model(export_args, model_path, imgsz):
    # Import export function from export.py
    from export import main as export_main

    export_args.weights = model_path
    export_args.imgsz = imgsz

    # Run the export
    export_main(export_args)

def main():

    args = parse_train_export_args()

    # Train the model and get the model path
    model_path, imgsz = train_model(args)

    print(model_path)
    print(imgsz)
    

    export_model(args, model_path, [args.imgsz, args.imgsz])
    
if __name__ == "__main__":
    main()
