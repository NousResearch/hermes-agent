#!/usr/bin/env python3
"""
Pipeline orchestrator — main entry point for watch-video skill.
Coordinates all stages with guaranteed cleanup.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

# Add parent directory to path for models import
sys.path.insert(0, str(Path(__file__).parent))
from models import PipelineOptions, VideoAsset, Frame, VisionResult, TimelineEntry, NotesResult, write_json, print_json


class TempWorkspace:
    """Manages temporary workspace with guaranteed cleanup."""

    def __init__(self, base_dir: Path, prefix: str = "watch-video-"):
        self.base_dir = base_dir
        self.prefix = prefix
        self.path: Optional[Path] = None

    def create(self) -> Path:
        """Create unique temporary workspace."""
        self.path = self.base_dir / f"{self.prefix}{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def cleanup(self) -> None:
        """Remove workspace and all contents."""
        if self.path and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None

    def __enter__(self) -> Path:
        return self.create()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def run_script(script: str, args: list[str], work_dir: Path, json_output: bool = True) -> dict:
    """Run a pipeline script and return parsed JSON output."""
    script_path = Path(__file__).parent / script
    cmd = [sys.executable, str(script_path)] + args
    if json_output:
        cmd.append("--json")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=work_dir, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"{script} failed (exit {result.returncode}): {result.stderr}")

    if json_output:
        return json.loads(result.stdout.strip())
    return {"stdout": result.stdout, "stderr": result.stderr}


def parse_time_arg(time_str: Optional[str]) -> Optional[float]:
    """Parse HH:MM:SS or seconds to float seconds."""
    if time_str is None:
        return None
    try:
        return float(time_str)
    except ValueError:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL for filename."""
    import re
    # YouTube patterns
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: hash of URL
    return str(hash(url))[:8]


def main():
    parser = argparse.ArgumentParser(description="Watch video pipeline — extract notes from video")
    parser.add_argument("url", help="Video URL (YouTube, Vimeo, etc.) or local file path")
    parser.add_argument("--start", help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("--end", help="End time (HH:MM:SS or seconds)")
    parser.add_argument("--max-frames", type=int, default=80, help="Max frames (cap 100)")
    parser.add_argument("--resolution", type=int, default=512, help="Frame width")
    parser.add_argument("--fps", type=float, help="Override fps (max 2)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--vision-model", default="qwen3-vl:4b", help="Vision model name")
    parser.add_argument("--vision-timeout", type=int, default=120, help="Vision timeout per batch")
    parser.add_argument("--vision-batch-size", type=int, default=8, help="Frames per vision batch")
    parser.add_argument("--out-dir", default="~/.hermes/watch-video-notes", help="Output directory")
    parser.add_argument("--work-dir", help="Working directory (default: auto temp)")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    # Validate max_frames
    max_frames = min(args.max_frames, 100)

    # Parse time args
    start_sec = parse_time_arg(args.start)
    end_sec = parse_time_arg(args.end)

    # Setup workspace
    if args.work_dir:
        work_base = Path(args.work_dir).expanduser()
    else:
        work_base = Path.home() / ".hermes" / "cache" / "watch-video"
    work_base.mkdir(parents=True, exist_ok=True)

    workspace = TempWorkspace(work_base)

    try:
        with workspace as work_dir:
            # Stage 1: Download video and captions
            if not args.json:
                print("Stage 1/5: Downloading video...", file=sys.stderr)

            download_result = run_script("download.py", [
                args.url,
                "--work-dir", str(work_dir),
            ], work_dir)

            video_asset = VideoAsset.from_dict(download_result)
            video_id = extract_video_id(args.url)
            video_asset.metadata["video_id"] = video_id
            video_asset.metadata["url"] = args.url

            # Stage 2: Extract frames
            if not args.json:
                print("Stage 2/5: Extracting frames...", file=sys.stderr)

            frames_result = run_script("extract_frames.py", [
                video_asset.path,
                "--work-dir", str(work_dir),
                "--max-frames", str(max_frames),
                "--resolution", str(args.resolution),
            ] + (["--fps", str(args.fps)] if args.fps else []) +
            (["--start", str(start_sec)] if start_sec is not None else []) +
            (["--end", str(end_sec)] if end_sec is not None else []),
            work_dir)

            frames = [Frame.from_dict(f) for f in frames_result.get("frames", [])]

            # Stage 3: Vision analysis
            if not args.json:
                print(f"Stage 3/5: Analyzing {len(frames)} frames with {args.vision_model}...", file=sys.stderr)

            # Save frames for vision script
            frames_json = work_dir / "frames.json"
            write_json({"frames": frames}, str(frames_json))

            vision_result = run_script("vision_analyze.py", [
                "--frames", str(frames_json),
                "--ollama-url", args.ollama_url,
                "--vision-model", args.vision_model,
                "--timeout", str(args.vision_timeout),
                "--batch-size", str(args.vision_batch_size),
            ], work_dir)

            vision_results = [VisionResult.from_dict(v) for v in vision_result.get("results", [])]

            # Stage 4: Merge timeline
            if not args.json:
                print("Stage 4/5: Merging timeline...", file=sys.stderr)

            # Save transcript
            transcript_json = work_dir / "transcript.json"
            write_json({"captions": video_asset.captions}, str(transcript_json))

            # Save vision
            vision_json = work_dir / "vision.json"
            write_json({"results": vision_results}, str(vision_json))

            timeline_result = run_script("merge_timeline.py", [
                "--transcript", str(transcript_json),
                "--vision", str(vision_json),
            ], work_dir)

            timeline = [TimelineEntry.from_dict(t) for t in timeline_result.get("timeline", [])]

            # Save timeline for write_notes.py
            write_json({"timeline": timeline}, str(work_dir / "timeline.json"))

            # Stage 5: Write notes
            if not args.json:
                print("Stage 5/5: Writing notes...", file=sys.stderr)

            # Save meta
            meta_json = work_dir / "meta.json"
            write_json(video_asset.metadata, str(meta_json))

            notes_result = run_script("write_notes.py", [
                "--timeline", str(work_dir / "timeline.json"),
                "--meta", str(meta_json),
                "--out-dir", args.out_dir,
            ], work_dir)

            notes = NotesResult.from_dict(notes_result)

            # Success
            if args.json:
                print_json(notes)
            else:
                print(notes.notes_path)

            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())