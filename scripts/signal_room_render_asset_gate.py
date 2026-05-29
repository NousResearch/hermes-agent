#!/usr/bin/env python3
"""Offline gate for a Signal Room rendered review MP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MIN_REVIEW_BYTES = 1024


def evaluate_render_asset(render_path: Path) -> dict:
    errors: list[str] = []
    exists = render_path.exists()
    size_bytes = render_path.stat().st_size if exists else 0
    header = render_path.read_bytes()[:32] if exists else b""

    if not exists:
        errors.append("missing render asset")
    if render_path.suffix.lower() != ".mp4":
        errors.append("render asset must be an .mp4 file")
    if exists and size_bytes < MIN_REVIEW_BYTES:
        errors.append("render asset is too small to be a usable review draft")
    if exists and b"ftyp" not in header:
        errors.append("render asset does not look like an MP4 container")

    return {
        "passed": not errors,
        "errors": errors,
        "render_path": str(render_path),
        "format": "mp4" if render_path.suffix.lower() == ".mp4" else render_path.suffix.lower().lstrip("."),
        "size_bytes": size_bytes,
        "review_only": True,
        "minimum_review_bytes": MIN_REVIEW_BYTES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_path", type=Path)
    parser.add_argument("--out", type=Path, help="write gate result JSON to this path")
    args = parser.parse_args()

    result = evaluate_render_asset(args.render_path)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
