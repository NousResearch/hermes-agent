#!/usr/bin/env python3
"""Detect shot cuts with ffmpeg scene detection and write a cut/shot manifest.

This upgrades shot tables from "粗略镜头切分" to evidence-backed cut lists.
Detected hard cuts are strong evidence; dissolves, slow transitions, and very
fast motion can still produce missed or false cuts, so reports must keep the
threshold caveat and spot-check against sampled frames.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

SCENE_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")
SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=(\d+(?:\.\d+)?)")
METADATA_FILENAME = "scene_metadata.txt"


def seconds_to_timecode(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    if total_s >= 3600:
        return f"{total_s // 3600:02d}:{(total_s % 3600) // 60:02d}:{total_s % 60:02d}.{ms:03d}"
    return f"{total_s // 60:02d}:{total_s % 60:02d}.{ms:03d}"


def parse_scene_metadata(text: str) -> list[dict]:
    """Parse ffmpeg metadata=print output into cut candidates."""
    cuts: list[dict] = []
    current_time: float | None = None
    for line in text.splitlines():
        time_match = SCENE_TIME_RE.search(line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
        score_match = SCENE_SCORE_RE.search(line)
        if score_match and current_time is not None:
            cuts.append({"time": current_time, "score": float(score_match.group(1))})
            current_time = None
    return cuts


def select_cuts(raw_cuts: list[dict], duration: float | None, min_shot_seconds: float) -> list[dict]:
    """Drop cuts outside the video range and merge cuts closer than min_shot_seconds."""
    cuts = sorted(
        (c for c in raw_cuts if c["time"] > 0 and (duration is None or c["time"] < duration)),
        key=lambda c: c["time"],
    )
    kept: list[dict] = []
    for cut in cuts:
        if kept and cut["time"] - kept[-1]["time"] < min_shot_seconds:
            if cut["score"] > kept[-1]["score"]:
                kept[-1] = cut
            continue
        kept.append(cut)
    return kept


def build_shots(cuts: list[dict], duration: float | None) -> list[dict]:
    boundaries = [0.0] + [c["time"] for c in cuts]
    if duration is not None and duration > boundaries[-1]:
        boundaries.append(duration)
    shots: list[dict] = []
    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        shots.append(
            {
                "shot": idx + 1,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "start_timecode": seconds_to_timecode(start),
                "end_timecode": seconds_to_timecode(end),
            }
        )
    return shots


def shot_stats(shots: list[dict]) -> dict:
    if not shots:
        return {"shot_count": 0}
    durations = [s["duration_seconds"] for s in shots]
    return {
        "shot_count": len(shots),
        "avg_shot_seconds": round(sum(durations) / len(durations), 3),
        "median_shot_seconds": round(statistics.median(durations), 3),
        "min_shot_seconds": round(min(durations), 3),
        "max_shot_seconds": round(max(durations), 3),
    }


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        cwd=str(cwd) if cwd else None,
    )


def probe_duration(video: Path) -> float | None:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ]
    )
    if result.returncode != 0:
        return None
    try:
        duration = json.loads(result.stdout).get("format", {}).get("duration")
        return float(duration) if duration else None
    except (json.JSONDecodeError, ValueError):
        return None


def run_scene_detection(video: Path, threshold: float) -> str:
    # metadata=print writes to a relative filename inside a temp cwd, which
    # avoids Windows drive-colon escaping inside the filtergraph.
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        result = run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                f"select=gt(scene\\,{threshold}),metadata=print:file={METADATA_FILENAME}",
                "-an",
                "-f",
                "null",
                "-",
            ],
            cwd=temp_dir,
        )
        if result.returncode != 0:
            raise SystemExit(f"ffmpeg scene detection failed:\n{result.stderr}")
        metadata_path = temp_dir / METADATA_FILENAME
        return metadata_path.read_text(encoding="utf-8", errors="replace") if metadata_path.exists() else ""


def write_csv(cuts: list[dict], shots: list[dict], out: Path) -> None:
    import csv

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["shot", "start_seconds", "end_seconds", "duration_seconds", "start_timecode", "end_timecode"])
        for shot in shots:
            writer.writerow(
                [
                    shot["shot"],
                    shot["start_seconds"],
                    shot["end_seconds"],
                    shot["duration_seconds"],
                    shot["start_timecode"],
                    shot["end_timecode"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect shot cuts via ffmpeg scene detection.")
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--out", type=Path, help="JSON output path; defaults to <video stem>_镜头切分.json beside the video")
    parser.add_argument("--csv-out", type=Path, help="optional shot-table CSV output")
    parser.add_argument("--threshold", type=float, default=0.3, help="scene-change score threshold, 0-1 (default 0.3)")
    parser.add_argument(
        "--min-shot-seconds",
        type=float,
        default=0.5,
        help="merge cuts closer than this many seconds, keeping the stronger cut (default 0.5)",
    )
    return parser.parse_args()


def main() -> int:
    import sys

    # Windows GBK consoles cannot print replacement chars from subprocess output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("missing required tool: ffmpeg/ffprobe")
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    duration = probe_duration(video)
    metadata_text = run_scene_detection(video, args.threshold)
    raw_cuts = parse_scene_metadata(metadata_text)
    cuts = select_cuts(raw_cuts, duration, args.min_shot_seconds)
    shots = build_shots(cuts, duration)

    result = {
        "source": str(video),
        "duration_seconds": round(duration, 3) if duration else None,
        "policy": {
            "method": "ffmpeg scene detection",
            "threshold": args.threshold,
            "min_shot_seconds": args.min_shot_seconds,
        },
        "limitations": [
            "硬切检测可靠；叠化、慢转场、黑场渐变可能漏切。",
            "快速运动、闪烁、强烈光效可能产生误切，需对照抽帧复核。",
        ],
        "raw_cut_candidates": len(raw_cuts),
        "cuts": [
            {
                "time_seconds": round(c["time"], 3),
                "timecode": seconds_to_timecode(c["time"]),
                "scene_score": round(c["score"], 4),
            }
            for c in cuts
        ],
        "shots": shots,
        "stats": shot_stats(shots),
    }

    out_path = args.out or video.with_name(f"{video.stem}_镜头切分.json")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(result["cuts"], shots, args.csv_out.resolve())

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out_path),
                "cut_count": len(cuts),
                **shot_stats(shots),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
