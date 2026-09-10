#!/usr/bin/env python3
"""TRELLIS.2 (microsoft/TRELLIS.2 HF Space) image-to-3D GLB generator.

Uses the caller's HF_TOKEN ZeroGPU quota. Single-file and batch modes.

Usage:
    python3 trellis_gen.py input.png -o out.glb --name goblin --preprocess
    python3 trellis_gen.py --batch images/ --out-root assets/ants --preprocess --keep-image

Requires: pip install "gradio_client>=1.0"
Env: HF_TOKEN must be set.
"""
import argparse
import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_qgates import gate_alpha_mask, gate_glb_geometry, apply_engine_preset

SPACE = "microsoft/TRELLIS.2"
MANIFEST_FIELDS = ["slug", "prompt", "image_path", "glb_path", "seed", "resolution", "status", "error"]

DEFAULTS = dict(
    resolution="512",
    decimation=300000,
    texture_size="2048",
    stage1=(7.5, 0.7, 12, 5),
    stage2=(7.5, 0.7, 12, 5),
    stage3=(1.0, 0.0, 12, 3),
)


def parse_stage(s):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("stage params must be 'guidance,rescale,steps,rescale_t'")
    # steps and rescale_t are ints
    return (parts[0], parts[1], int(parts[2]), int(parts[3]))


def get_client():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN is not set. Create a token at https://huggingface.co/settings/tokens")
    try:
        from gradio_client import Client
    except ImportError:
        sys.exit("ERROR: gradio_client not installed. Run: pip install 'gradio_client>=1.0'")
    return Client(SPACE, token=token)


def gen_one(client, image_path, out_glb, name, args):
    """Full chain for one image. Returns (seed, error_or_none)."""
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    from gradio_client import handle_file
    img = handle_file(str(image_path))
    if args.preprocess:
        print(f"[{name}] preprocess_image ...", flush=True)
        res = client.predict(img, api_name="/preprocess_image")
        img = res[0] if isinstance(res, (list, tuple)) else res
        if isinstance(img, dict):
            img = img.get("path") or img.get("url")
        img = handle_file(img)

    flat = [v for st in (args.stage1, args.stage2, args.stage3) for v in st]
    print(f"[{name}] image_to_3d seed={seed} res={args.resolution} ...", flush=True)
    client.predict(img, seed, args.resolution, *flat, api_name="/image_to_3d")

    # extract_glb is the slow step; retry with backoff (gen is cached per session)
    last_err = None
    for attempt in range(4):
        try:
            print(f"[{name}] extract_glb (attempt {attempt + 1}) ...", flush=True)
            res = client.predict(args.decimation, args.texture_size, api_name="/extract_glb")
            glb_src = None
            for item in (res if isinstance(res, (list, tuple)) else [res]):
                if isinstance(item, dict):
                    item = item.get("path") or item.get("url")
                if isinstance(item, str) and item.lower().endswith(".glb"):
                    glb_src = item
                    break
            if not glb_src:
                raise RuntimeError(f"no .glb in extract result: {res!r}")
            Path(out_glb).parent.mkdir(parents=True, exist_ok=True)
            if glb_src.startswith("http"):
                import urllib.request
                req = urllib.request.Request(glb_src, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
                with urllib.request.urlopen(req) as r, open(out_glb, "wb") as f:
                    shutil.copyfileobj(r, f)
            else:
                shutil.copyfile(glb_src, out_glb)
            print(f"[{name}] OK -> {out_glb}", flush=True)
            return seed, None
        except Exception as e:
            last_err = e
            wait = 15 * (attempt + 1)
            print(f"[{name}] extract failed: {e}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    return seed, str(last_err)


def append_manifest(root, row):
    path = Path(root) / "manifest.csv"
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="input image (single mode)")
    ap.add_argument("-o", "--output", help="output .glb path (single mode)")
    ap.add_argument("--name", default=None, help="asset slug (default: input filename stem)")
    ap.add_argument("--batch", help="directory of images to convert (batch mode)")
    ap.add_argument("--out-root", help="batch output root (creates glb/ and images/ + manifest.csv)")
    ap.add_argument("--prompts-file", help="optional txt file (one prompt per line, order matches sorted images) recorded in manifest")
    ap.add_argument("--preprocess", action="store_true", help="run /preprocess_image (alpha-mask) before generation")
    ap.add_argument("--keep-image", action="store_true", help="copy the input image into out-root/images/ in batch mode")
    ap.add_argument("--seed", type=int, default=None, help="fixed seed (default: random per asset)")
    ap.add_argument("--resolution", default=DEFAULTS["resolution"], choices=["512", "1024", "1536"])
    ap.add_argument("--decimation", type=int, default=DEFAULTS["decimation"])
    ap.add_argument("--texture-size", type=int, default=2048, choices=[1024, 2048, 4096])
    ap.add_argument("--stage1", type=parse_stage, default=DEFAULTS["stage1"])
    ap.add_argument("--stage2", type=parse_stage, default=DEFAULTS["stage2"])
    ap.add_argument("--stage3", type=parse_stage, default=DEFAULTS["stage3"])
    ap.add_argument("--target", choices=["godot-mobile", "godot", "unity", "blender", "film"],
                    help="engine preset: sets --decimation/--texture-size unless explicitly overridden")
    ap.add_argument("--resume", action="store_true",
                    help="batch mode: skip assets whose output .glb already exists")
    ap.add_argument("--skip-gates", action="store_true", help="disable quality gates")
    args = ap.parse_args()
    apply_engine_preset(args)

    client = get_client()

    if args.batch:
        out_root = Path(args.out_root or "assets_out")
        glb_dir = out_root / "glb"
        img_dir = out_root / "images"
        glb_dir.mkdir(parents=True, exist_ok=True)
        prompts = []
        if args.prompts_file:
            prompts = [l.strip() for l in open(args.prompts_file) if l.strip()]
        images = sorted(p for p in Path(args.batch).iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if not images:
            sys.exit(f"no images found in {args.batch}")
        ok = 0
        for i, img_path in enumerate(images):
            stem = img_path.stem
            # avoid double prefix when inputs are already NN_ named
            slug = stem if stem[:2].isdigit() and stem[2:3] == "_" else f"{i + 1:02d}_{stem}"
            prompt = prompts[i] if i < len(prompts) else ""
            out_glb = glb_dir / f"{slug}.glb"
            if args.resume and out_glb.exists():
                print(f"[{slug}] exists, skipping (--resume)", flush=True)
                ok += 1
                continue
            if not args.skip_gates and not args.preprocess:
                # gate local masks only; --preprocess masks server-side instead
                ok_g, msg = gate_alpha_mask(img_path)
                print(f"[{slug}] gate(alpha): {msg}", flush=True)
                if not ok_g:
                    append_manifest(out_root, dict(slug=slug, prompt=prompt, image_path=str(img_path),
                                                   glb_path="", seed="", resolution=args.resolution,
                                                   status="gate_failed", error=f"alpha: {msg}"))
                    continue
            if args.keep_image:
                img_dir.mkdir(exist_ok=True)
                shutil.copyfile(img_path, img_dir / f"{slug}{img_path.suffix}")
            seed, err = gen_one(client, img_path, out_glb, slug, args)
            if not err and not args.skip_gates:
                ok_g, msg = gate_glb_geometry(out_glb)
                print(f"[{slug}] gate(glb): {msg}", flush=True)
                if not ok_g:
                    err = f"glb_gate: {msg}"
            append_manifest(out_root, dict(
                slug=slug, prompt=prompt,
                image_path=str(img_path), glb_path=str(out_glb) if not err else "",
                seed=seed, resolution=args.resolution,
                status="ok" if not err else "error", error=err or ""))
            ok += 0 if err else 1
        print(f"done: {ok}/{len(images)} succeeded. Manifest: {out_root / 'manifest.csv'}")
        sys.exit(0 if ok == len(images) else 1)

    if not args.input or not args.output:
        ap.error("single mode requires INPUT and -o OUTPUT (or use --batch)")
    name = args.name or Path(args.input).stem
    seed, err = gen_one(client, args.input, args.output, name, args)
    if err:
        sys.exit(f"FAILED: {err}")


if __name__ == "__main__":
    main()
