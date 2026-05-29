#!/usr/bin/env python3
"""Validate Signal Room adult character rig-acting frame exports.

This is the deterministic gate after the Blender/Moho pose render pass. It
checks that a candidate folder has the required transparent PNG pose/mouth
frames, that the frames are large enough for the next proof assembly, and that
the review manifest stays non-public.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

REQUIRED_FRAMES = (
    "neutral_read.png",
    "bill_shock.png",
    "skeptical_point.png",
    "look_to_machine.png",
    "lean_weight_shift.png",
    "mouth_closed.png",
    "mouth_a.png",
    "mouth_o.png",
)

MIN_HEIGHT = 1400
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_png_info(path: Path) -> dict[str, int | bool]:
    data = path.read_bytes()
    if len(data) < 33 or not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    if data[12:16] != b"IHDR":
        raise ValueError("missing PNG IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "has_alpha": color_type in (4, 6),
    }


def read_manifest(candidate_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = candidate_dir / "rig_pass_manifest.json"
    if not manifest_path.exists():
        return None, ["missing rig_pass_manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return None, [f"invalid rig_pass_manifest.json: {exc.msg}"]

    errors: list[str] = []
    if manifest.get("public_release") is not False:
        errors.append("manifest public_release must be false for review gate")
    if not manifest.get("license_status"):
        errors.append("manifest license_status is required")
    if not manifest.get("render_tool"):
        errors.append("manifest render_tool is required")
    return manifest, errors


def evaluate_candidate_dir(candidate_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    frames: dict[str, Any] = {}

    for frame_name in REQUIRED_FRAMES:
        frame_path = candidate_dir / frame_name
        if not frame_path.exists():
            errors.append(f"missing required frame: {frame_name}")
            continue
        try:
            info = read_png_info(frame_path)
        except ValueError as exc:
            errors.append(f"{frame_name} {exc}")
            continue
        frames[frame_name] = info
        if info["height"] < MIN_HEIGHT:
            errors.append(f"{frame_name} height {info['height']} below minimum {MIN_HEIGHT}")
        if not info["has_alpha"]:
            errors.append(f"{frame_name} is not RGBA/alpha PNG")

    manifest, manifest_errors = read_manifest(candidate_dir)
    errors.extend(manifest_errors)

    return {
        "candidate": candidate_dir.name,
        "candidate_dir": str(candidate_dir),
        "passed": not errors,
        "errors": errors,
        "frames": frames,
        "manifest": manifest,
    }


def find_candidate_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "rig_pass_manifest.json").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "rig_pass_manifest.json").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="candidate frame directory or parent character_frames directory")
    parser.add_argument("--out", type=Path, help="write scorecard JSON to this path")
    args = parser.parse_args()

    candidate_dirs = find_candidate_dirs(args.path)
    if not candidate_dirs:
        result: dict[str, Any] = {
            "passed": False,
            "errors": [f"no candidate directories with rig_pass_manifest.json found under {args.path}"],
            "candidates": [],
        }
    else:
        candidates = [evaluate_candidate_dir(path) for path in candidate_dirs]
        result = {
            "passed": all(candidate["passed"] for candidate in candidates),
            "candidates": candidates,
        }

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
