#!/usr/bin/env python3
"""sample frames from a video and create contact sheets for director-lapian.

requires ffmpeg/ffprobe and pillow. this script is intentionally lightweight:
it extracts frames at a fixed interval and records metadata, but does not claim
professional shot detection or frame-accurate edit lists.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    Image = None
    ImageDraw = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing required tool: {name}")


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid time value: {value}") from exc
    raise argparse.ArgumentTypeError(f"invalid time value: {value}")


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def ffprobe_metadata(video: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    duration = None
    if data.get("format", {}).get("duration"):
        duration = float(data["format"]["duration"])
    elif video_stream.get("duration"):
        duration = float(video_stream["duration"])
    return {
        "input": str(video),
        "duration_seconds": duration,
        "duration_timecode": format_time(duration or 0),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "audio_stream_count": len(audio_streams),
    }


def existing_sampled_frames(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []


def extract_frames(video: Path, frames_dir: Path, interval: float, start: float | None, duration: float | None, max_frames: int | None) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "frame_%06d.jpg"
    fps = 1.0 / interval
    vf = f"fps={fps}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd.extend(["-ss", str(start)])
    cmd.extend(["-i", str(video)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend(["-vf", vf, "-q:v", "2"])
    if max_frames:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append(str(output_pattern))
    result = run_command(cmd)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed:\n{result.stderr}")
    return sorted(frames_dir.glob("frame_*.jpg"))


def write_manifest(frames: list[Path], out_dir: Path, interval: float, start: float | None) -> None:
    base = start or 0.0
    rows = []
    for idx, frame in enumerate(frames):
        timestamp = base + idx * interval
        rows.append({
            "index": idx + 1,
            "timestamp_seconds": round(timestamp, 3),
            "timestamp_timecode": format_time(timestamp),
            "file": str(frame),
        })
    (out_dir / "frames_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "frames_manifest.csv").open("w", encoding="utf-8") as f:
        f.write("index,timestamp_seconds,timestamp_timecode,file\n")
        for row in rows:
            f.write(f"{row['index']},{row['timestamp_seconds']},{row['timestamp_timecode']},{row['file']}\n")


def create_contact_sheets(frames: list[Path], out_dir: Path, cols: int, thumb_width: int, max_per_sheet: int, interval: float, start: float | None) -> list[Path]:
    if Image is None:
        raise SystemExit(f"pillow is required for contact sheets: {PIL_IMPORT_ERROR}")
    if not frames:
        return []
    sheets_dir = out_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: list[Path] = []
    for sheet_index, start_idx in enumerate(range(0, len(frames), max_per_sheet), start=1):
        chunk = frames[start_idx:start_idx + max_per_sheet]
        thumbs = []
        label_h = 28
        for frame in chunk:
            img = Image.open(frame).convert("RGB")
            ratio = thumb_width / img.width
            thumb_height = max(1, int(img.height * ratio))
            img = img.resize((thumb_width, thumb_height))
            thumbs.append(img)
        rows = math.ceil(len(thumbs) / cols)
        thumb_height = max(img.height for img in thumbs)
        sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_height + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, img in enumerate(thumbs):
            global_index = start_idx + i
            x = (i % cols) * thumb_width
            y = (i // cols) * (thumb_height + label_h)
            sheet.paste(img, (x, y))
            timestamp = (start or 0.0) + global_index * interval
            label = f"#{global_index + 1}  {format_time(timestamp)}  {frames[global_index].stem}"
            draw.text((x + 5, y + thumb_height + 6), label, fill=(0, 0, 0))
        path = sheets_dir / f"contact_sheet_{sheet_index:02d}.jpg"
        sheet.save(path, quality=90)
        sheet_paths.append(path)
    return sheet_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="sample frames and create contact sheets for director-lapian")
    parser.add_argument("video", type=Path, help="input video path")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between sampled frames, e.g. 2, 1, or 0.5")
    parser.add_argument("--start", type=str, default=None, help="optional start time, seconds or hh:mm:ss")
    parser.add_argument("--duration", type=str, default=None, help="optional duration, seconds or hh:mm:ss")
    parser.add_argument("--max-frames", type=int, default=None, help="optional cap on extracted frames")
    parser.add_argument("--contact-sheet", action="store_true", help="create contact sheet jpg files")
    parser.add_argument("--reuse-existing-frames", action="store_true", help="reuse existing frame_*.jpg files instead of extracting new frames")
    parser.add_argument("--sheet-cols", type=int, default=5, help="contact sheet columns")
    parser.add_argument("--thumb-width", type=int, default=320, help="thumbnail width in pixels")
    parser.add_argument("--max-per-sheet", type=int, default=60, help="maximum thumbnails per contact sheet")
    args = parser.parse_args()

    require_tool("ffmpeg")
    require_tool("ffprobe")

    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than 0")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    start = parse_time(args.start)
    duration = parse_time(args.duration)

    existing_frames = existing_sampled_frames(frames_dir)
    if existing_frames and not args.reuse_existing_frames:
        raise SystemExit(
            "output frames already exist. choose a new --out-dir, manually clear the frames folder, "
            "or pass --reuse-existing-frames to build manifests/contact sheets from the existing frames."
        )

    metadata = ffprobe_metadata(video)
    metadata.update({
        "sampling_interval_seconds": args.interval,
        "sampling_start_seconds": start,
        "sampling_duration_seconds": duration,
    })
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if existing_frames and args.reuse_existing_frames:
        frames = existing_frames
    else:
        frames = extract_frames(video, frames_dir, args.interval, start, duration, args.max_frames)
    write_manifest(frames, out_dir, args.interval, start)

    sheets = []
    if args.contact_sheet:
        sheets = create_contact_sheets(frames, out_dir, args.sheet_cols, args.thumb_width, args.max_per_sheet, args.interval, start)

    summary = {
        "video": str(video),
        "out_dir": str(out_dir),
        "frame_count": len(frames),
        "first_frame": str(frames[0]) if frames else None,
        "last_frame": str(frames[-1]) if frames else None,
        "last_timecode": format_time((start or 0.0) + (len(frames) - 1) * args.interval) if frames else None,
        "contact_sheets": [str(p) for p in sheets],
        "metadata_file": str(out_dir / "metadata.json"),
        "manifest_json": str(out_dir / "frames_manifest.json"),
        "manifest_csv": str(out_dir / "frames_manifest.csv"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
