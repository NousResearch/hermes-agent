#!/usr/bin/env python3
"""Sample Signal Room proof frames from a rendered review MP4."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


WIDTH = 1080
HEIGHT = 1920
COMPOSITION_ID = "fee-machine-v2-review"


Runner = Callable[[list[str]], None]


def read_frame_plan(plan_path: Path) -> dict[str, Any]:
    return json.loads(plan_path.read_text())


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


def sample_render_frames(
    plan_path: Path,
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

    plan = read_frame_plan(plan_path)
    frames = []
    for frame in plan.get("frames", []):
        frame_id = str(frame["id"])
        sample_time = float(frame["sample_time"])
        filename = f"{frame_id}.png"
        out_path = out_dir / filename
        runner(build_ffmpeg_command(render_path, sample_time, out_path))
        frames.append(
            {
                "id": frame_id,
                "sample_time": sample_time,
                "beat": frame.get("beat"),
                "filename": filename,
                "placeholder": False,
                "source": "sampled from rendered HyperFrames draft",
            }
        )

    manifest = {
        "status": "review-only",
        "public_release": False,
        "source_composition": COMPOSITION_ID,
        "source_render": str(render_path),
        "render_tool": "ffmpeg",
        "frame_size": {"width": WIDTH, "height": HEIGHT},
        "frames": frames,
    }
    (out_dir / "proof_frame_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--render", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace existing output directory")
    args = parser.parse_args()

    manifest = sample_render_frames(args.plan, args.render, args.out, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
