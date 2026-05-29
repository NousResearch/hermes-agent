#!/usr/bin/env python3
"""Create first-frame hook candidates from a Signal Room review MP4."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


WIDTH = 1080
HEIGHT = 1920
Runner = Callable[[list[str]], None]

HOOK_CANDIDATES = (
    {
        "id": "cold_open_bill",
        "sample_time": 0.35,
        "question": "who is affected and what object starts the story?",
    },
    {
        "id": "human_read",
        "sample_time": 1.25,
        "question": "does the character read as a person with a practical problem?",
    },
    {
        "id": "pre_split_tension",
        "sample_time": 2.15,
        "question": "is there enough visual tension before the fee split begins?",
    },
)


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_ffmpeg_command(render_path: Path, sample_time: float, out_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{sample_time:g}",
        "-i",
        str(render_path),
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out_path),
    ]


def render_review_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Signal Room First-Frame Review",
        "",
        f"Source render: `{manifest['source_render']}`",
        "",
        "Reject the opening if the first frame cannot answer who is affected, what object starts the story, and why the viewer should keep watching.",
        "",
        "## Candidates",
    ]
    for candidate in manifest["candidates"]:
        lines.extend(
            [
                f"### {candidate['id']}",
                f"- File: `{candidate['filename']}`",
                f"- Sample: {candidate['sample_time']}s",
                f"- Question: {candidate['question']}",
                "- Decision: [ ] Keep / [ ] Revise / [ ] Reject",
                "- Notes:",
                "",
            ]
        )
    return "\n".join(lines)


def create_first_frame_pack(
    render_path: Path,
    out_dir: Path,
    *,
    force: bool = False,
    runner: Runner = run_ffmpeg,
) -> dict[str, Any]:
    if out_dir.exists():
        if not force:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    candidates = []
    for candidate in HOOK_CANDIDATES:
        filename = f"{candidate['id']}.png"
        out_path = out_dir / filename
        runner(build_ffmpeg_command(render_path, float(candidate["sample_time"]), out_path))
        candidates.append(
            {
                **candidate,
                "filename": filename,
                "placeholder": False,
            }
        )

    manifest = {
        "status": "review-only",
        "public_release": False,
        "source_render": str(render_path),
        "frame_size": {"width": WIDTH, "height": HEIGHT},
        "candidates": candidates,
    }
    (out_dir / "first_frame_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "first_frame_review.md").write_text(render_review_markdown(manifest) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = create_first_frame_pack(args.render, args.out, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
