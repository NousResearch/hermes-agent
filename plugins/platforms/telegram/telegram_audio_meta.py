"""Audio facts the Telegram media sends carry: duration, and the music-player fields of ``sendAudio``."""

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=False)
            if proc.returncode == 0:
                return _coerce_duration_seconds(proc.stdout.strip())
    except Exception:
        pass
    return None

def audio_player_fields(path: str) -> dict[str, str]:
    """``{"performer", "title", "thumbnail_path"}`` read from ``path``'s own tags; missing keys are omitted.

    ``thumbnail_path`` is a temporary 320px JPEG of the embedded cover art the caller must delete.
    Best effort and blocking (ffprobe/ffmpeg): run it in ``asyncio.to_thread``."""
    if not shutil.which("ffprobe"):
        return {}
    fields: dict[str, str] = {}
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=artist,title:stream=codec_type",
             "-of", "json", path],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=False)
        blob = json.loads(proc.stdout or "{}") if proc.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        logger.debug("[Telegram] audio tag probe failed for %s", path, exc_info=True)
        return {}
    tags = {k.lower(): str(v).strip() for k, v in ((blob.get("format") or {}).get("tags") or {}).items()}
    for tag, field in (("artist", "performer"), ("title", "title")):
        if tags.get(tag):
            fields[field] = tags[tag][:256]
    has_cover = any(s.get("codec_type") == "video" for s in blob.get("streams") or [])
    if has_cover:
        cover = _extract_cover_jpeg(path)
        if cover:
            fields["thumbnail_path"] = cover
    return fields


def _extract_cover_jpeg(path: str) -> Optional[str]:
    """A temporary JPEG of ``path``'s embedded cover art (see :func:`fit_thumbnail_jpeg`), or None."""
    return fit_thumbnail_jpeg(path, prefix="hermes-tg-cover-")


def fit_thumbnail_jpeg(path: str, *, seek: Optional[int] = None, prefix: str = "hermes-tg-thumb-",
                       timeout: int = 20) -> Optional[str]:
    """A temporary JPEG of one frame of ``path`` (at ``seek`` seconds) fitted inside 320x320, or None.

    Telegram limits a ``thumbnail`` to 320px on BOTH sides, so ``scale=320:-2`` would send portrait
    art 320 wide and thousands tall. Every file operation stays inside the best-effort boundary: a
    full or unwritable temp dir costs the thumbnail, never the send. The caller deletes the file."""
    if not shutil.which("ffmpeg"):
        return None
    out = None
    try:
        fd, out = tempfile.mkstemp(suffix=".jpg", prefix=prefix)
        os.close(fd)
        seek_args = ["-ss", str(seek)] if seek is not None else []
        ok = subprocess.run(
            ["ffmpeg", "-y", *seek_args, "-i", path, "-an", "-frames:v", "1",
             "-vf", "scale=320:320:force_original_aspect_ratio=decrease", "-q:v", "6", out],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False).returncode == 0
        if ok and os.path.getsize(out):
            return out
    except (OSError, subprocess.SubprocessError):
        logger.debug("[Telegram] thumbnail extraction failed for %s", path, exc_info=True)
    if out:
        with contextlib.suppress(OSError):
            os.remove(out)
    return None
