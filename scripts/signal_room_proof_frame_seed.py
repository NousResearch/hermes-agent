#!/usr/bin/env python3
"""Create review-only placeholder proof frames from a retention frame plan."""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any


WIDTH = 1080
HEIGHT = 1920
COMPOSITION_ID = "fee-machine-v2-review"


def read_frame_plan(plan_path: Path) -> dict[str, Any]:
    return json.loads(plan_path.read_text())


def png_bytes(*, width: int, height: int, shade: int) -> bytes:
    raw = b"".join(b"\x00" + bytes([shade, shade, shade]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def create_proof_frame_seed(plan_path: Path, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    if out_dir.exists():
        if not force:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    plan = read_frame_plan(plan_path)
    frames = []
    for index, frame in enumerate(plan.get("frames", []), start=1):
        frame_id = str(frame["id"])
        shade = max(20, 235 - index * 31)
        (out_dir / f"{frame_id}.png").write_bytes(png_bytes(width=WIDTH, height=HEIGHT, shade=shade))
        frames.append(
            {
                "id": frame_id,
                "sample_time": frame.get("sample_time"),
                "filename": f"{frame_id}.png",
                "placeholder": True,
                "replacement_required": "HyperFrames sampled proof frame export",
            }
        )

    manifest = {
        "status": "review-only",
        "public_release": False,
        "source_composition": COMPOSITION_ID,
        "render_tool": "placeholder seed; replace with HyperFrames frame export",
        "frame_size": {"width": WIDTH, "height": HEIGHT},
        "frames": frames,
    }
    (out_dir / "proof_frame_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace existing output directory")
    args = parser.parse_args()

    manifest = create_proof_frame_seed(args.plan, args.out, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
