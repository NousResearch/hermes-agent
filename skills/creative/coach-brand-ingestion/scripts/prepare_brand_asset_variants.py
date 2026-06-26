#!/usr/bin/env python3
"""Prepare PPTX-ready PNG variants for a coach Brand Design Doc asset bundle.

Input: a directory containing approved identity assets such as logo.svg, icon.svg,
wordmark.svg, or existing PNGs.

Output: transparent/high-resolution PNG variants where possible:
  logo@2x.png, logo@4x.png, icon@2x.png, icon@4x.png, wordmark@2x.png, wordmark@4x.png

macOS fallback uses qlmanage because it is available on Mini3-style hosts without
requiring Cairo. This helper does not approve assets; it only renders variants for
assets already accepted by the Brand Design Document workflow.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

IDENTITY_STEMS = ("logo", "icon", "wordmark", "symbol", "mark")
SCALES = {"2x": 2048, "4x": 4096}


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def find_assets(asset_dir: Path) -> List[Path]:
    assets: List[Path] = []
    for ext in ("*.svg", "*.png", "*.jpg", "*.jpeg", "*.webp"):
        assets.extend(asset_dir.glob(ext))
    return sorted(p for p in assets if any(stem in p.stem.lower() for stem in IDENTITY_STEMS))


def canonical_stem(path: Path) -> str:
    name = path.stem.lower()
    for stem in ("wordmark", "logo", "icon", "symbol", "mark"):
        if stem in name:
            return "icon" if stem in {"symbol", "mark"} else stem
    return path.stem


def render_svg_with_qlmanage(svg: Path, out_png: Path, size: int) -> Dict[str, str]:
    qlmanage = shutil.which("qlmanage") or "/usr/bin/qlmanage"
    if not Path(qlmanage).exists():
        return {"ok": False, "error": "qlmanage_unavailable"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        proc = run([qlmanage, "-t", "-s", str(size), "-o", str(tmp), str(svg)])
        candidates = list(tmp.glob("*.png"))
        if proc.returncode != 0 or not candidates:
            return {"ok": False, "error": "qlmanage_failed", "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}
        shutil.copy2(candidates[0], out_png)
    return {"ok": True, "method": "qlmanage", "size": str(size)}


def copy_existing_raster(src: Path, out_png: Path) -> Dict[str, str]:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, out_png)
        return {"ok": True, "method": "copy_png"}
    # Keep non-PNG rasters visible but do not silently re-encode without Pillow.
    try:
        from PIL import Image
        im = Image.open(src)
        im.save(out_png)
        return {"ok": True, "method": "pillow_raster_to_png"}
    except Exception as e:  # pragma: no cover - environment-dependent
        return {"ok": False, "error": f"raster_conversion_failed:{type(e).__name__}:{e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Create PPTX-ready PNG variants for approved brand identity assets.")
    ap.add_argument("asset_dir", type=Path, help="Directory containing approved brand assets")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory; defaults to asset_dir")
    ap.add_argument("--manifest", type=Path, default=None, help="Manifest JSON path; defaults to out-dir/asset-variants-manifest.json")
    args = ap.parse_args()

    asset_dir = args.asset_dir.expanduser().resolve()
    out_dir = (args.out_dir or asset_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest or (out_dir / "asset-variants-manifest.json")).expanduser().resolve()

    manifest = {
        "asset_dir": str(asset_dir),
        "out_dir": str(out_dir),
        "note": "Variants generated for assets already approved by the Brand Design Document workflow; this script does not approve assets.",
        "variants": [],
        "blockers": [],
    }

    assets = find_assets(asset_dir)
    if not assets:
        manifest["blockers"].append("no_identity_assets_found")

    for src in assets:
        stem = canonical_stem(src)
        source_entry = {"source": str(src), "source_sha256": sha256(src), "stem": stem, "outputs": []}
        for label, size in SCALES.items():
            out_png = out_dir / f"{stem}@{label}.png"
            if src.suffix.lower() == ".svg":
                result = render_svg_with_qlmanage(src, out_png, size)
            else:
                result = copy_existing_raster(src, out_png)
            if result.get("ok") and out_png.exists() and out_png.stat().st_size > 0:
                source_entry["outputs"].append({"path": str(out_png), "sha256": sha256(out_png), **result})
            else:
                manifest["blockers"].append({"source": str(src), "target": str(out_png), **result})
        manifest["variants"].append(source_entry)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0 if not manifest["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
