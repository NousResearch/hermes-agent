"""Best-effort media probing for Telegram sends: audio length, video geometry, and a poster frame.

Telegram renders a voice note or video without explicit metadata as 0:00 / square / blank, so the
adapter probes the file first. Every helper is blocking and best-effort (None / {} when unreadable):
callers run them via ``asyncio.to_thread``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.telegram.adapter")


def _coerce_duration_seconds(value: Any) -> Optional[int]:
    """Round a raw length to whole positive seconds, or None if unusable."""
    try:
        secs = round(float(value))
    except (TypeError, ValueError):
        return None
    return secs if secs > 0 else None


def _probe_voice_duration_seconds(path: str) -> Optional[int]:
    """Best-effort whole-second audio length (wave → mutagen → ffprobe; None if unreadable).

    Telegram renders long clips as 0:00 without an explicit duration. Blocking: use ``to_thread``."""
    if os.path.splitext(path)[1].lower() == ".wav":
        try:
            import wave
            with wave.open(path, "rb") as wf:
                rate = wf.getframerate() or 0
                secs = _coerce_duration_seconds(wf.getnframes() / float(rate)) if rate else None
            if secs is not None:
                return secs
        except Exception:
            pass
    try:
        import mutagen
        secs = _coerce_duration_seconds(getattr(getattr(mutagen.File(path), "info", None), "length", None))
        if secs is not None:
            return secs
    except Exception:
        pass
    try:
        import shutil
        import subprocess
        if shutil.which("ffprobe"):
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
            if proc.returncode == 0:
                return _coerce_duration_seconds(proc.stdout.strip())
    except Exception:
        pass
    return None


def _probe_video_geometry(path: str) -> dict[str, int]:
    """``{"width", "height", "duration"}`` for a local video; ``{}`` when ffprobe can't read it.

    Telegram runs its own video processing only for uploads under roughly 10 MB; above that it
    stores the file as an unprocessed ``320x320`` video with ``duration=0``, so the message must
    carry the real geometry or clients draw a square tile for any aspect ratio.
    """
    try:
        import shutil
        import subprocess
        if not shutil.which("ffprobe"):
            return {}
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "json", path],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        if proc.returncode != 0:
            return {}
        blob = json.loads(proc.stdout or "{}")
        streams = blob.get("streams") or []
        if not streams:
            return {}
        geometry = {"width": int(streams[0]["width"]), "height": int(streams[0]["height"])}
        duration = _coerce_duration_seconds((blob.get("format") or {}).get("duration"))
        if duration:
            geometry["duration"] = duration
        return geometry
    except Exception:
        logger.debug("[Telegram] video geometry probe failed for %s", path, exc_info=True)
        return {}


def _video_thumbnail_jpeg(path: str, duration: Optional[int]) -> Optional[str]:
    """Write a 320px-wide JPEG frame for Telegram's ``thumbnail`` field; None on failure.

    Telegram keeps a supplied thumbnail for the uploads it did not process itself — without one the
    chat shows a square placeholder tile until the video is opened.
    """
    out = None
    try:
        import shutil
        import subprocess
        import tempfile
        if not shutil.which("ffmpeg"):
            return None
        seek = max(1, int((duration or 3) * 0.25))
        fd, out = tempfile.mkstemp(suffix=".jpg", prefix="hermes-tg-thumb-")
        os.close(fd)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seek), "-i", path, "-frames:v", "1",
             "-vf", "scale=320:-2", "-q:v", "6", out],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if proc.returncode != 0 or not os.path.getsize(out):
            with contextlib.suppress(OSError):
                os.remove(out)
            return None
        return out
    except Exception:
        logger.debug("[Telegram] video thumbnail extraction failed for %s", path, exc_info=True)
        if out:
            with contextlib.suppress(OSError):
                os.remove(out)
        return None
