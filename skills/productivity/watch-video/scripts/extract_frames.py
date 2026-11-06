#!/usr/bin/env python3
"""
Frame extractor — uses ffmpeg to extract frames at adaptive intervals.
"""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for models import
sys.path.insert(0, str(Path(__file__).parent))
from models import Frame, VideoAsset, write_json, print_json


def calculate_fps(duration: float, max_frames: int, target_fps: Optional[float] = None) -> float:
    """Calculate adaptive fps based on duration and max frames."""
    if target_fps is not None:
        return min(target_fps, 2.0)  # Cap at 2 fps

    # Adaptive intervals matching upstream claude-video behavior
    if duration < 300:  # < 5 min
        fps = 1.0
    elif duration < 1800:  # 5-30 min
        fps = 0.2  # 1 frame per 5 seconds
    elif duration < 3600:  # 30-60 min
        fps = 0.1  # 1 frame per 10 seconds
    else:  # > 60 min
        fps = 0.05  # 1 frame per 20 seconds

    # Ensure we don't exceed max_frames
    estimated_frames = duration * fps
    if estimated_frames > max_frames:
        fps = max_frames / duration

    return min(fps, 2.0)


def parse_time_arg(time_str: Optional[str], duration: float) -> Optional[float]:
    """Parse HH:MM:SS or seconds to float seconds."""
    if time_str is None:
        return None
    try:
        # Try as seconds first
        return float(time_str)
    except ValueError:
        # Parse HH:MM:SS or MM:SS
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])


def extract_frames(
    video_path: Path,
    work_dir: Path,
    max_frames: int = 80,
    resolution: int = 512,
    fps: Optional[float] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> list[Frame]:
    """Extract frames using ffmpeg. Returns list of Frame objects."""

    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg filter
    vf_parts = []

    # Time range
    if start is not None or end is not None:
        ss = start or 0
        to = end if end is not None else ""
        vf_parts.append(f"trim=start={ss}:end={to}" if to else f"trim=start={ss}")

    # FPS filter
    vf_parts.append(f"fps={fps}")

    # Scale to target resolution (maintain aspect ratio)
    vf_parts.append(f"scale={resolution}:-1")

    vf = ",".join(vf_parts)

    # Output pattern
    output_pattern = str(frames_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "2",  # High quality JPEG
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr}")

    # Collect frames
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    frames = []

    for idx, frame_path in enumerate(frame_files):
        # Calculate timestamp
        start_sec = start if start is not None else 0.0
        fps_val = fps if fps is not None and fps > 0 else 1.0
        timestamp = start_sec + (idx / fps_val)

        # Get dimensions
        width, height = get_frame_dimensions(frame_path)

        frames.append(Frame(
            path=str(frame_path),
            timestamp=timestamp,
            index=idx,
            width=width,
            height=height,
        ))

    return frames


def get_frame_dimensions(frame_path: Path) -> tuple[int, int]:
    """Get frame dimensions using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(frame_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    if result.returncode != 0:
        return 512, 288  # Default fallback
    parts = result.stdout.strip().split(",")
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 512, 288


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--work-dir", required=True, help="Working directory")
    parser.add_argument("--max-frames", type=int, default=80, help="Max frames (cap 100)")
    parser.add_argument("--resolution", type=int, default=512, help="Frame width")
    parser.add_argument("--fps", type=float, help="Override fps (max 2)")
    parser.add_argument("--start", help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", help="End time (HH:MM:SS or seconds)")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    # Cap max_frames at 100
    max_frames = min(args.max_frames, 100)

    video_path = Path(args.video)
    work_dir = Path(args.work_dir)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 2

    try:
        # Get video duration for adaptive fps
        duration = get_duration(video_path)

        # Parse time args
        start_sec = parse_time_arg(args.start, duration)
        end_sec = parse_time_arg(args.end, duration)

        # Calculate effective duration for fps calculation
        effective_duration = duration
        if start_sec is not None and end_sec is not None:
            effective_duration = end_sec - start_sec
        elif start_sec is not None:
            effective_duration = duration - start_sec
        elif end_sec is not None:
            effective_duration = end_sec

        # Calculate fps
        calc_fps = calculate_fps(effective_duration, max_frames, args.fps)

        # Extract frames
        frames = extract_frames(
            video_path=video_path,
            work_dir=work_dir,
            max_frames=max_frames,
            resolution=args.resolution,
            fps=calc_fps,
            start=start_sec,
            end=end_sec,
        )

        # Enforce max_frames cap (ffmpeg might produce slightly more)
        if len(frames) > max_frames:
            # Evenly sample to max_frames
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]
            # Re-index
            for i, f in enumerate(frames):
                f.index = i

        if args.json:
            print_json({"frames": frames, "fps_used": calc_fps, "duration": duration})
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def get_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


if __name__ == "__main__":
    sys.exit(main())