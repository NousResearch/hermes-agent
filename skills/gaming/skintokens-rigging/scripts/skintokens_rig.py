#!/usr/bin/env python3
"""VAST-AI/SkinTokens (HF Space) auto-rigger: static mesh -> rigged GLB.

Uses the caller's HF_TOKEN ZeroGPU quota. Single-file and batch modes.

Usage:
    python3 skintokens_rig.py model.glb -o rigged/model_rigged.glb
    python3 skintokens_rig.py --batch assets/ants/glb/ -o assets/ants/rigged/ --preserve-texture-scale

Requires: pip install "gradio_client>=1.0"
Env: HF_TOKEN must be set.
"""
import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from pipeline_qgates import gate_glb_rig
except ImportError:
    # standalone use outside the trellis skill dir: try sibling skill location
    sys.path.insert(0, str(Path.home() / ".hermes/skills/gaming/trellis-image-to-3d/scripts"))
    from pipeline_qgates import gate_glb_rig

SPACE = "VAST-AI/SkinTokens"
CHECKPOINT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
MANIFEST_FIELDS = ["input", "output", "status", "error"]


def get_client():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN is not set. Create one at https://huggingface.co/settings/tokens")
    try:
        from gradio_client import Client
    except ImportError:
        sys.exit("ERROR: gradio_client not installed. Run: pip install 'gradio_client>=1.0'")
    return Client(SPACE, token=token)


def download(url, dest):
    import urllib.request
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def rig_one(client, model_path, out_glb, args):
    """Rig a single mesh. Returns error string or None."""
    from gradio_client import handle_file
    payload = [
        [handle_file(str(model_path))],  # models (list of files)
        args.top_k,
        args.top_p,
        args.temperature,
        args.rep_penalty,
        args.beams,
        args.use_existing_skeleton,
        args.preserve_texture_scale,
        args.voxel_postprocess,
        CHECKPOINT,
        "None",
    ]
    last_err = None
    for attempt in range(3):
        try:
            # auto-escalate: repeated "No output files" failures are rescued by
            # voxel post-processing (confirmed on translucent/glossy/rocky inputs)
            if attempt >= 1 and not payload[8] and "No output files" in str(last_err or ""):
                print(f"[{Path(model_path).stem}] auto-enabling voxel post-processing", flush=True)
                payload[8] = True
            print(f"[{Path(model_path).stem}] run_gradio (attempt {attempt + 1}) ...", flush=True)
            res = client.predict(*payload, api_name="/run_gradio")
            status, glb = (res if isinstance(res, (list, tuple)) else (None, res))[:2]
            # output may be a single file or a list of files (one per input model)
            if isinstance(glb, (list, tuple)):
                glb = glb[0] if glb else None
            if isinstance(glb, dict):
                glb = glb.get("path") or glb.get("url") or glb.get("value")
            if not (isinstance(glb, str) and glb.lower().endswith(".glb")):
                raise RuntimeError(f"no rigged .glb in result (status={status!r}, out={glb!r})")
            Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
            if glb.startswith("http"):
                download(glb, out_glb)
            else:
                shutil.copyfile(glb, out_glb)
            print(f"[{Path(model_path).stem}] OK -> {out_glb}", flush=True)
            return None
        except Exception as e:
            last_err = e
            wait = 20 * (attempt + 1)
            print(f"[{Path(model_path).stem}] failed: {e}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    return str(last_err)


def append_manifest(out_dir, row):
    path = Path(out_dir) / "rig_manifest.csv"
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="input mesh (.glb/.obj/.fbx)")
    ap.add_argument("-o", "--output", required=True, help="output .glb path (single) or directory (batch)")
    ap.add_argument("--batch", help="directory of meshes to rig")
    ap.add_argument("--top-k", type=float, default=5)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rep-penalty", type=float, default=2.0)
    ap.add_argument("--beams", type=float, default=10)
    ap.add_argument("--use-existing-skeleton", action="store_true")
    ap.add_argument("--preserve-texture-scale", action="store_true",
                    help="keep original textures/scale (recommended for TRELLIS.2 outputs)")
    ap.add_argument("--voxel-postprocess", action="store_true",
                    help="extra skin-weight cleanup; try for thin limbs")
    ap.add_argument("--resume", action="store_true",
                    help="batch mode: skip meshes whose output .glb already exists")
    ap.add_argument("--skip-gates", action="store_true", help="disable rig quality gate")
    args = ap.parse_args()

    client = get_client()

    if args.batch:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        meshes = sorted(p for p in Path(args.batch).iterdir()
                        if p.suffix.lower() in (".glb", ".obj", ".fbx"))
        if not meshes:
            sys.exit(f"no meshes found in {args.batch}")
        ok = 0
        for m in meshes:
            out_glb = out_dir / f"{m.stem}_rigged.glb"
            if args.resume and out_glb.exists():
                print(f"[{m.stem}] exists, skipping (--resume)", flush=True)
                ok += 1
                continue
            err = rig_one(client, m, out_glb, args)
            if not err and not args.skip_gates:
                ok_g, msg = gate_glb_rig(out_glb)
                print(f"[{m.stem}] gate(rig): {msg}", flush=True)
                if not ok_g:
                    err = f"rig_gate: {msg}"
            append_manifest(out_dir, dict(input=str(m), output=str(out_glb) if not err else "",
                                          status="ok" if not err else "error", error=err or ""))
            ok += 0 if err else 1
        print(f"done: {ok}/{len(meshes)} succeeded. Manifest: {out_dir / 'rig_manifest.csv'}")
        sys.exit(0 if ok == len(meshes) else 1)

    if not args.input:
        ap.error("single mode requires INPUT (or use --batch)")
    err = rig_one(client, args.input, args.output, args)
    if err:
        sys.exit(f"FAILED: {err}")


if __name__ == "__main__":
    main()
