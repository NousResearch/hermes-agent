#!/usr/bin/env python3
"""Mux Signal Room review SFX assets onto a rendered draft MP4."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


Runner = Callable[[list[str]], None]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _manifest_assets(asset_dir: Path) -> dict[str, str]:
    manifest = read_json(asset_dir / "audio_asset_manifest.json")
    return {str(asset["cue_id"]): str(asset["filename"]) for asset in manifest.get("assets", [])}


def _cue_entries(cue_sheet: Path, asset_dir: Path) -> list[dict[str, Any]]:
    sheet = read_json(cue_sheet)
    manifest_assets = _manifest_assets(asset_dir)
    entries = []
    for cue in sheet.get("cues", []):
        cue_id = str(cue["id"])
        filename = manifest_assets.get(cue_id, f"{cue_id}.wav")
        entries.append(
            {
                "id": cue_id,
                "start": float(cue.get("start", 0.0)),
                "path": asset_dir / filename,
            }
        )
    return entries


def build_mux_command(cue_sheet: Path, asset_dir: Path, render_path: Path, out_path: Path) -> tuple[list[str], dict[str, Any]]:
    sheet = read_json(cue_sheet)
    entries = _cue_entries(cue_sheet, asset_dir)
    command = ["ffmpeg", "-y", "-i", str(render_path)]
    for entry in entries:
        command.extend(["-i", str(entry["path"])])

    filters = []
    delayed_labels = []
    for index, entry in enumerate(entries, start=1):
        label = f"a{index}"
        delay_ms = int(round(entry["start"] * 1000))
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
        delayed_labels.append(f"[{label}]")
    filters.append(
        "".join(delayed_labels)
        + f"amix=inputs={len(entries)}:duration=longest:normalize=0,"
        + "alimiter=limit=0.92[aout]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(out_path),
        ]
    )
    metadata = {
        "cue_count": len(entries),
        "duration_seconds": sheet.get("duration_seconds"),
        "render_path": str(render_path),
        "asset_dir": str(asset_dir),
        "output": str(out_path),
    }
    return command, metadata


def create_audio_muxed_review(
    cue_sheet: Path,
    asset_dir: Path,
    render_path: Path,
    out_path: Path,
    *,
    scorecard_out: Path | None = None,
    runner: Runner = run_ffmpeg,
) -> dict[str, Any]:
    command, metadata = build_mux_command(cue_sheet, asset_dir, render_path, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner(command)
    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    errors = []
    if not out_path.exists():
        errors.append("missing audio-muxed review MP4")
    elif size_bytes < 1024:
        errors.append("audio-muxed review MP4 is too small")

    result = {
        "passed": not errors,
        "errors": errors,
        "output": str(out_path),
        "size_bytes": size_bytes,
        "review_only": True,
        **metadata,
    }
    if scorecard_out:
        scorecard_out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cue-sheet", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--render", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scorecard-out", type=Path)
    args = parser.parse_args()

    result = create_audio_muxed_review(
        args.cue_sheet,
        args.assets,
        args.render,
        args.out,
        scorecard_out=args.scorecard_out,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
