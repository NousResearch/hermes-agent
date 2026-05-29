#!/usr/bin/env python3
"""Validate Signal Room proof frames against a retention frame plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUIRED_WIDTH = 1080
REQUIRED_HEIGHT = 1920


def read_png_info(path: Path) -> dict[str, int]:
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
    }


def read_frame_plan(plan_path: Path) -> dict[str, Any]:
    return json.loads(plan_path.read_text())


def required_frame_ids(plan_path: Path) -> list[str]:
    plan = read_frame_plan(plan_path)
    return [str(frame["id"]) for frame in plan.get("frames", []) if frame.get("id")]


def evaluate_choreography_frame_coverage(
    choreography_path: Path,
    frame_ids: list[str],
    frame_infos: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    coverage: list[dict[str, Any]] = []
    choreography = json.loads(choreography_path.read_text())
    valid_frame_ids = set(frame_ids)
    for beat in choreography.get("beats", []):
        if not isinstance(beat, dict):
            continue
        beat_id = str(beat.get("id", ""))
        review_frame = beat.get("review_frame")
        if not review_frame:
            errors.append(f"choreography beat {beat_id} missing review_frame")
            continue
        review_frame = str(review_frame)
        frame_name = f"{review_frame}.png"
        if review_frame not in valid_frame_ids:
            errors.append(f"choreography beat {beat_id} review_frame {review_frame} is not in retention plan")
        frame_present = frame_name in frame_infos
        frame_valid = False
        if frame_present:
            info = frame_infos[frame_name]
            frame_valid = info.get("width") == REQUIRED_WIDTH and info.get("height") == REQUIRED_HEIGHT
        else:
            errors.append(f"choreography beat {beat_id} review frame missing: {frame_name}")
        coverage.append(
            {
                "beat_id": beat_id,
                "review_frame": review_frame,
                "frame_name": frame_name,
                "frame_present": frame_present,
                "frame_valid": frame_valid,
            }
        )
    return coverage, errors


def read_manifest(frames_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = frames_dir / "proof_frame_manifest.json"
    if not manifest_path.exists():
        return None, ["missing proof_frame_manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return None, [f"invalid proof_frame_manifest.json: {exc.msg}"]

    errors: list[str] = []
    if manifest.get("public_release") is not False:
        errors.append("manifest public_release must be false for review gate")
    if not manifest.get("source_composition"):
        errors.append("manifest source_composition is required")
    if not manifest.get("render_tool"):
        errors.append("manifest render_tool is required")
    return manifest, errors


def evaluate_retention_frames(
    plan_path: Path,
    frames_dir: Path,
    *,
    choreography_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    frame_infos: dict[str, Any] = {}
    frame_hashes: list[str] = []
    frame_ids = required_frame_ids(plan_path)

    for frame_id in frame_ids:
        frame_name = f"{frame_id}.png"
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            errors.append(f"missing required frame: {frame_name}")
            continue
        try:
            info = read_png_info(frame_path)
        except ValueError as exc:
            errors.append(f"{frame_name} {exc}")
            continue
        frame_infos[frame_name] = info
        if info["width"] != REQUIRED_WIDTH or info["height"] != REQUIRED_HEIGHT:
            errors.append(
                f"{frame_name} dimensions {info['width']}x{info['height']} "
                f"must be {REQUIRED_WIDTH}x{REQUIRED_HEIGHT}"
            )
        frame_hashes.append(hashlib.sha256(frame_path.read_bytes()).hexdigest())

    if len(frame_hashes) > 1 and len(set(frame_hashes)) != len(frame_hashes):
        errors.append("frame images must be visually distinct; duplicate file content detected")

    manifest, manifest_errors = read_manifest(frames_dir)
    errors.extend(manifest_errors)
    choreography_frame_coverage: list[dict[str, Any]] = []
    if choreography_path is not None:
        if choreography_path.exists():
            choreography_frame_coverage, choreography_errors = evaluate_choreography_frame_coverage(
                choreography_path,
                frame_ids,
                frame_infos,
            )
            errors.extend(choreography_errors)
        else:
            errors.append(f"missing choreography file: {choreography_path}")

    return {
        "passed": not errors,
        "errors": errors,
        "plan": str(plan_path),
        "frames_dir": str(frames_dir),
        "required_frame_ids": frame_ids,
        "frames": frame_infos,
        "manifest": manifest,
        "choreography_frame_coverage": choreography_frame_coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="retention_frame_plan.json from scaffold")
    parser.add_argument("--frames", required=True, type=Path, help="directory containing sampled proof PNG frames")
    parser.add_argument("--choreography", type=Path, help="scene_choreography.json from scaffold")
    parser.add_argument("--out", type=Path, help="write gate result JSON to this path")
    args = parser.parse_args()

    result = evaluate_retention_frames(args.plan, args.frames, choreography_path=args.choreography)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
