#!/usr/bin/env python3
"""Select a sparser frame set for lapian reports while preserving evidence.

The evidence layer can stay at 1fps, while the report layer defaults to a
2-second cadence and keeps important seconds from subtitles, audio peaks, and
manual dense ranges.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


def second_to_timecode(second: int) -> str:
    return f"{second // 60:02d}:{second % 60:02d}"


def parse_time(value: str) -> int:
    value = value.strip()
    if re.fullmatch(r"\d+", value):
        return int(value)
    parts = value.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(float(parts[1]))
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    raise ValueError(f"unsupported time value: {value}")


def parse_range(value: str) -> tuple[int, int]:
    if "-" not in value:
        start = parse_time(value)
        return start, start
    start_s, end_s = value.split("-", 1)
    start = parse_time(start_s)
    end = parse_time(end_s)
    if end < start:
        start, end = end, start
    return start, end


def load_frame_manifest(manifest_path: Path | None = None, frames_dir: Path | None = None) -> list[dict]:
    frames: list[dict] = []
    if manifest_path and manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("frames") or data.get("items") or []
        for idx, item in enumerate(data, 1):
            second = int(round(float(item.get("timestamp_seconds", idx - 1))))
            frames.append(
                {
                    "index": int(item.get("index", idx)),
                    "second": second,
                    "timecode": item.get("timestamp_timecode") or second_to_timecode(second),
                    "file": item.get("file") or item.get("path") or "",
                }
            )
    elif frames_dir and frames_dir.exists():
        image_files = sorted((frames_dir / "frames").glob("frame_*.jpg"))
        if not image_files:
            image_files = sorted(frames_dir.glob("frame_*.jpg"))
        for idx, path in enumerate(image_files, 1):
            second = idx - 1
            frames.append({"index": idx, "second": second, "timecode": second_to_timecode(second), "file": str(path)})
    if not frames:
        raise SystemExit("no frames found; pass --manifest or --frames-dir")
    return frames


def add_reason(reasons: dict[int, set[str]], second: int, reason: str) -> None:
    reasons.setdefault(second, set()).add(reason)


def add_subtitle_seconds(reasons: dict[int, set[str]], subtitle_path: Path | None) -> None:
    if not subtitle_path or not subtitle_path.exists():
        return
    if subtitle_path.suffix.lower() == ".jsonl":
        segments = [json.loads(line) for line in subtitle_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        data = json.loads(subtitle_path.read_text(encoding="utf-8-sig"))
        segments = data.get("segments", data if isinstance(data, list) else [])
    for segment in segments:
        if "start" not in segment:
            continue
        start = int(math.floor(float(segment["start"])))
        add_reason(reasons, start, "subtitle")


def add_audio_peak_seconds(reasons: dict[int, set[str]], levels_path: Path | None, top_audio_peaks: int) -> None:
    if not levels_path or not levels_path.exists() or top_audio_peaks <= 0:
        return
    rows = []
    with levels_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                second = int(float(row.get("second") or row.get("time") or row.get("start") or 0))
                peak = float(row.get("peak") or 0)
                rms = float(row.get("rms") or row.get("db") or 0)
            except ValueError:
                continue
            rows.append((peak, rms, second))
    for _, _, second in sorted(rows, reverse=True)[:top_audio_peaks]:
        add_reason(reasons, second, "audio_peak")


def add_dense_ranges(reasons: dict[int, set[str]], dense_ranges: Iterable[str], dense_interval: int) -> None:
    for item in dense_ranges:
        start, end = parse_range(item)
        for second in range(start, end + 1, max(1, dense_interval)):
            add_reason(reasons, second, "dense_range")


def build_selection_plan(
    manifest_path: Path | None = None,
    frames_dir: Path | None = None,
    report_interval: int = 2,
    subtitle_path: Path | None = None,
    levels_path: Path | None = None,
    top_audio_peaks: int = 20,
    dense_ranges: Iterable[str] = (),
    dense_interval: int = 1,
) -> dict:
    frames = load_frame_manifest(manifest_path, frames_dir)
    by_second = {item["second"]: item for item in frames}
    min_second = min(by_second)
    max_second = max(by_second)
    reasons: dict[int, set[str]] = {}

    add_reason(reasons, min_second, "first_frame")
    add_reason(reasons, max_second, "last_frame")
    for second in range(min_second, max_second + 1):
        if (second - min_second) % max(1, report_interval) == 0:
            add_reason(reasons, second, f"{report_interval}s_cadence")

    add_subtitle_seconds(reasons, subtitle_path)
    add_audio_peak_seconds(reasons, levels_path, top_audio_peaks)
    add_dense_ranges(reasons, dense_ranges, dense_interval)

    selected = []
    for second in sorted(reasons):
        frame = by_second.get(second)
        if not frame:
            continue
        selected.append(
            {
                "index": frame["index"],
                "second": second,
                "timecode": frame["timecode"],
                "file": frame["file"],
                "reasons": sorted(reasons[second]),
            }
        )

    return {
        "policy": {
            "evidence_interval_seconds": 1,
            "report_interval_seconds": report_interval,
            "dense_interval_seconds": dense_interval,
            "top_audio_peaks": top_audio_peaks,
        },
        "total_frames": len(frames),
        "selected_count": len(selected),
        "reduction_ratio": round(len(selected) / len(frames), 4),
        "selected_frames": selected,
    }


def write_csv(plan: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "second", "timecode", "file", "reasons"])
        for item in plan["selected_frames"]:
            writer.writerow([item["index"], item["second"], item["timecode"], item["file"], ";".join(item["reasons"])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select report display frames from 1fps lapian evidence.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--subtitle-json", type=Path)
    parser.add_argument("--levels-csv", type=Path)
    parser.add_argument("--report-interval", type=int, default=2)
    parser.add_argument("--top-audio-peaks", type=int, default=20)
    parser.add_argument("--dense-range", action="append", default=[], help="manual dense range, e.g. 03:10-03:25")
    parser.add_argument("--dense-interval", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_selection_plan(
        manifest_path=args.manifest,
        frames_dir=args.frames_dir,
        report_interval=args.report_interval,
        subtitle_path=args.subtitle_json,
        levels_path=args.levels_csv,
        top_audio_peaks=args.top_audio_peaks,
        dense_ranges=args.dense_range,
        dense_interval=args.dense_interval,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        write_csv(plan, args.csv_out)
    print(json.dumps({k: plan[k] for k in ["total_frames", "selected_count", "reduction_ratio"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
