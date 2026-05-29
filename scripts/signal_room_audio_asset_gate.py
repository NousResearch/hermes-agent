#!/usr/bin/env python3
"""Validate Signal Room review audio assets against an audio cue sheet."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


def read_cue_sheet(cue_sheet: Path) -> dict[str, Any]:
    return json.loads(cue_sheet.read_text())


def required_cue_ids(cue_sheet: Path) -> list[str]:
    data = read_cue_sheet(cue_sheet)
    return [str(cue["id"]) for cue in data.get("cues", []) if cue.get("id")]


def read_wav_info(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    if data[12:16] != b"fmt ":
        raise ValueError("missing fmt chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack("<HHIIHH", data[20:36])
    return {
        "audio_format": audio_format,
        "channels": channels,
        "sample_rate": sample_rate,
        "byte_rate": byte_rate,
        "block_align": block_align,
        "bits_per_sample": bits_per_sample,
    }


def read_manifest(asset_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    manifest_path = asset_dir / "audio_asset_manifest.json"
    if not manifest_path.exists():
        return None, ["missing audio_asset_manifest.json"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return None, [f"invalid audio_asset_manifest.json: {exc.msg}"]

    errors: list[str] = []
    if manifest.get("status") != "review-only":
        errors.append("manifest status must be review-only")
    if manifest.get("public_release") is not False:
        errors.append("manifest public_release must be false for review gate")
    return manifest, errors


def evaluate_audio_assets(cue_sheet: Path, asset_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    cue_ids = required_cue_ids(cue_sheet)
    manifest, manifest_errors = read_manifest(asset_dir)
    errors.extend(manifest_errors)

    manifest_assets = {}
    if manifest:
        manifest_assets = {str(asset.get("cue_id")): str(asset.get("filename")) for asset in manifest.get("assets", [])}

    assets: dict[str, Any] = {}
    for cue_id in cue_ids:
        filename = manifest_assets.get(cue_id, f"{cue_id}.wav")
        path = asset_dir / filename
        if not path.exists():
            errors.append(f"missing required audio asset: {filename}")
            continue
        try:
            assets[filename] = read_wav_info(path)
        except ValueError as exc:
            errors.append(f"{filename} {exc}")

    return {
        "passed": not errors,
        "errors": errors,
        "cue_sheet": str(cue_sheet),
        "asset_dir": str(asset_dir),
        "required_cue_ids": cue_ids,
        "assets": assets,
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue-sheet", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="write gate result JSON to this path")
    args = parser.parse_args()

    result = evaluate_audio_assets(args.cue_sheet, args.assets)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
