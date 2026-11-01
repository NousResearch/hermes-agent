#!/usr/bin/env python3
"""
Video downloader — uses yt-dlp to download video and extract captions.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add parent directory to path for models import
sys.path.insert(0, str(Path(__file__).parent))
from models import VideoAsset, TranscriptSegment, write_json, print_json


def parse_vtt(vtt_path: Path) -> list[TranscriptSegment]:
    """Parse WebVTT file into TranscriptSegment list."""
    segments = []
    current_start = 0.0
    current_end = 0.0
    current_text = []

    with open(vtt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line == "WEBVTT":
                continue

            # Timestamp line: 00:00:01.000 --> 00:00:04.000
            if "-->" in line:
                if current_text:
                    segments.append(TranscriptSegment(
                        start=current_start,
                        end=current_end,
                        text=" ".join(current_text).strip(),
                    ))
                parts = line.split("-->")
                current_start = parse_timestamp(parts[0].strip())
                current_end = parse_timestamp(parts[1].strip())
                current_text = []
            elif not line.startswith("NOTE") and not line.startswith("STYLE"):
                current_text.append(line)

        # Last segment
        if current_start is not None and current_text:
            segments.append(TranscriptSegment(
                start=current_start,
                end=current_end,
                text=" ".join(current_text).strip(),
            ))

    return segments


def parse_timestamp(ts: str) -> float:
    """Parse SS.mmm, MM:SS.mmm or HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 1:
        return float(parts[0])
    else:
        return 0.0
    return int(h) * 3600 + int(m) * 60 + float(s)


def run_yt_dlp(args: list[str], work_dir: Path) -> subprocess.CompletedProcess:
    """Run yt-dlp with common settings."""
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-color",
        "--ignore-errors",
        "--no-playlist",
        "-P", str(work_dir),
    ] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)


def get_video_info(url: str, work_dir: Path) -> dict:
    """Get video metadata without downloading."""
    result = run_yt_dlp(["--dump-json", url], work_dir)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp info failed: {result.stderr}")
    return json.loads(result.stdout)


def download_captions(url: str, work_dir: Path) -> tuple[list[TranscriptSegment], Optional[Path]]:
    """Download captions only (no video). Returns (segments, vtt_path)."""
    # Try auto-generated captions first, then manual
    for sub_lang in ["en", "en-US", "en-GB", "auto"]:
        run_yt_dlp([
            "--write-auto-sub",
            "--sub-lang", sub_lang,
            "--sub-format", "vtt",
            "--skip-download",
            url,
        ], work_dir)

        # Find downloaded VTT file
        vtt_files = list(work_dir.glob("*.vtt"))
        if vtt_files:
            segments = parse_vtt(vtt_files[0])
            if segments:
                return segments, vtt_files[0]

    return [], None


def download_video(url: str, work_dir: Path) -> Path:
    """Download full video. Returns video file path."""
    result = run_yt_dlp([
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", "video.%(ext)s",
        url,
    ], work_dir)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {result.stderr}")

    # Find downloaded video
    video_files = list(work_dir.glob("video.*"))
    if not video_files:
        raise RuntimeError("Video file not found after download")
    return video_files[0]


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


def main():
    parser = argparse.ArgumentParser(description="Download video and extract captions")
    parser.add_argument("url", help="Video URL or local file path")
    parser.add_argument("--work-dir", required=True, help="Working directory")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    url = args.url

    # Handle local file
    if Path(url).exists():
        video_path = Path(url).resolve()
        duration = get_duration(video_path)
        asset = VideoAsset(
            path=str(video_path),
            title=video_path.stem,
            duration=duration,
            captions=[],
            metadata={"local_file": True},
        )
        if args.json:
            print_json(asset)
        return 0

    try:
        # Get metadata first
        info = get_video_info(url, work_dir)
        title = info.get("title", "unknown")
        duration = info.get("duration", 0)

        # Try captions
        captions, _vtt_path = download_captions(url, work_dir)

        # Download video
        video_path = download_video(url, work_dir)

        # Verify duration
        actual_duration = get_duration(video_path)

        asset = VideoAsset(
            path=str(video_path),
            title=title,
            duration=actual_duration or duration,
            captions=captions,
            metadata=info,
        )

        if args.json:
            print_json(asset)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())