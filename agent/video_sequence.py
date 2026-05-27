"""
Video Sequence — Segment Planning & Media Utilities
====================================================

Core module for long-video continuous generation.

Provides:
  - VideoSequenceRequest / VideoSegmentPlan dataclasses
  - plan_segments() — split total duration respecting provider caps
  - materialize_video() — download/copy a video to the local sequence cache
  - extract_last_frame() — pull the last frame from a video via ffmpeg
  - image_file_to_data_url() — base64-encode an image file as a data URL
  - concat_videos() — concatenate mp4 files via ffmpeg re-encode

Cache layout: $HERMES_HOME/cache/videos/sequences/<sequence_id>/
"""

from __future__ import annotations

import base64
import math
import mimetypes
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _sequence_cache_dir(sequence_id: str) -> Path:
    """Return $HERMES_HOME/cache/videos/sequences/<sequence_id>/, creating parents."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "cache" / "videos" / "sequences" / sequence_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VideoSegmentPlan:
    """A single planned video segment in a sequence."""

    index: int
    start_seconds: float
    duration: int  # seconds, clamped to provider [min_duration, max_duration]
    prompt_hint: str = ""


@dataclass
class VideoSequenceRequest:
    """Top-level request for a long video composed of multiple segments."""

    total_duration: int  # target total seconds
    segment_duration: int  # preferred seconds per segment
    prompt: str
    sequence_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    provider_name: str = ""

    # Filled in by plan_segments()
    segments: List[VideoSegmentPlan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------


def plan_segments(
    total_duration: int,
    segment_duration: int,
    provider_caps: Dict[str, Any],
) -> List[VideoSegmentPlan]:
    """Split *total_duration* into segments respecting provider capability limits.

    Args:
        total_duration: Target total video length in seconds.
        segment_duration: Preferred duration per segment in seconds.
        provider_caps: Dict returned by ``VideoGenProvider.capabilities()``.
            Reads ``min_duration`` (default 1) and ``max_duration`` (default 10).

    Returns:
        Ordered list of :class:`VideoSegmentPlan` covering the full duration.
        The last segment may be shorter than ``segment_duration`` when the
        total is not evenly divisible, but never below ``min_duration``.
    """
    min_dur: int = int(provider_caps.get("min_duration", 1))
    max_dur: int = int(provider_caps.get("max_duration", 10))

    # Clamp the preferred segment duration to provider limits.
    clamped = max(min_dur, min(segment_duration, max_dur))

    n_segments = math.ceil(total_duration / clamped)
    plans: List[VideoSegmentPlan] = []

    remaining = total_duration
    for i in range(n_segments):
        dur = min(clamped, remaining)
        # Never emit a segment shorter than the provider minimum.
        if dur < min_dur:
            dur = min_dur
        plans.append(
            VideoSegmentPlan(
                index=i,
                start_seconds=i * clamped,
                duration=dur,
            )
        )
        remaining -= dur
        if remaining <= 0:
            break

    return plans


# ---------------------------------------------------------------------------
# Media utilities
# ---------------------------------------------------------------------------


def materialize_video(
    url_or_path: str,
    *,
    sequence_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """Download or copy a video into the local sequence cache.

    Args:
        url_or_path: HTTP(S) URL or absolute/relative filesystem path.
        sequence_id: Cache sub-directory key. Created as a random UUID when omitted.
        filename: Override the saved filename. Derived from the source when omitted.

    Returns:
        Absolute :class:`Path` to the cached video file.
    """
    sid = sequence_id or uuid.uuid4().hex
    cache_dir = _sequence_cache_dir(sid)

    if url_or_path.startswith(("http://", "https://")):
        if not filename:
            url_path = url_or_path.split("?")[0].rstrip("/")
            filename = url_path.split("/")[-1] or f"video_{uuid.uuid4().hex[:8]}.mp4"
        dest = cache_dir / filename
        urllib.request.urlretrieve(url_or_path, dest)
    else:
        src = Path(url_or_path)
        dest = cache_dir / (filename or src.name)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    return dest


def extract_last_frame(video_path: Path) -> Path:
    """Extract the last frame of *video_path* as a PNG via ffmpeg.

    The output is written to the same directory as the source video,
    named ``<stem>_last_frame.png``.

    Raises:
        FileNotFoundError: If ffmpeg is not available on PATH.
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero status.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH — install it to use extract_last_frame()"
        )

    video_path = Path(video_path)
    out_path = video_path.parent / f"{video_path.stem}_last_frame.png"

    # Probe the stream duration so we can seek close to the end.
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    seek_to: Optional[float] = None
    for line in probe_result.stdout.splitlines():
        line = line.strip()
        try:
            dur = float(line)
            seek_to = max(0.0, dur - 0.1)
            break
        except ValueError:
            continue

    if seek_to is not None:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(seek_to),
            "-i", str(video_path),
            "-vframes", "1",
            str(out_path),
        ]
    else:
        # Fallback when duration probe fails: use sseof to grab last frame.
        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1",
            "-i", str(video_path),
            "-vframes", "1",
            "-update", "1",
            str(out_path),
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def image_file_to_data_url(path: Path) -> str:
    """Read an image file and return a base64 data URL.

    Args:
        path: Path to the image file (PNG, JPEG, WebP, …).

    Returns:
        ``data:<mime>;base64,<encoded>`` string.
    """
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"  # safe default for ffmpeg-extracted frames
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def concat_videos(
    paths: List[Path],
    output_path: Path,
) -> Path:
    """Concatenate a list of mp4 files into a single output via ffmpeg re-encode.

    Uses the ``concat`` demuxer with a temporary file-list, then re-encodes
    to H.264/AAC for a clean, seekable output.

    Args:
        paths: Ordered list of source video paths.
        output_path: Destination path for the concatenated mp4.

    Returns:
        Absolute :class:`Path` to the output file.

    Raises:
        FileNotFoundError: If ffmpeg is not available on PATH.
        ValueError: If *paths* is empty.
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero status.
    """
    if not paths:
        raise ValueError("concat_videos() requires at least one input path")

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH — install it to use concat_videos()"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as flist:
        for p in paths:
            safe = str(Path(p).resolve()).replace("'", "'\\''")
            flist.write(f"file '{safe}'\n")
        flist_path = flist.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", flist_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        Path(flist_path).unlink(missing_ok=True)

    return output_path.resolve()
