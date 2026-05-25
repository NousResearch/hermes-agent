"""WeChat voice-bubble conversion utilities.

Transcodes TTS output (OGG/Opus) into SILK format for native WeChat voice
bubbles.  Also provides duration detection for both formats.

Requires:
  - ``pilk`` (optional, installed via ``pip install pilk``)
  - ``ffmpeg`` on PATH (used for OGG → PCM decoding)

If ``pilk`` is unavailable, :func:`ogg_to_silk` returns *None* so callers
can fall back to file-attachment delivery.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HAS_PILK = False
try:
    import pilk  # type: ignore[import-untyped]

    _HAS_PILK = True
except ImportError:
    pilk = None  # type: ignore[assignment]


def _has_pilk() -> bool:
    return _HAS_PILK


def ogg_to_silk(ogg_path: str, *, sample_rate: int = 24000) -> Optional[str]:
    """Convert an OGG/Opus audio file to SILK format.

    Pipeline: OGG → PCM (ffmpeg) → SILK (pilk)

    Returns the path to the generated ``.silk`` file, or *None* if
    conversion is not possible (missing pilk or ffmpeg failure).
    """
    if not _has_pilk():
        logger.debug("pilk not installed — cannot convert to SILK")
        return None

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.debug("ffmpeg not found on PATH — cannot convert OGG to PCM")
        return None

    ogg_file = Path(ogg_path)
    if not ogg_file.is_file():
        logger.warning("ogg_to_silk: source file does not exist: %s", ogg_path)
        return None

    tmp_dir = tempfile.mkdtemp(prefix="hermes-silk-")
    pcm_path = os.path.join(tmp_dir, "audio.pcm")
    silk_path = os.path.join(tmp_dir, f"{ogg_file.stem}.silk")

    try:
        # Step 1: OGG → raw PCM via ffmpeg
        import subprocess

        cmd = [
            ffmpeg,
            "-y",             # overwrite output
            "-i", str(ogg_file),
            "-f", "s16le",    # raw 16-bit little-endian PCM
            "-acodec", "pcm_s16le",
            "-ac", "1",       # mono
            "-ar", str(sample_rate),
            pcm_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "ffmpeg OGG→PCM failed (rc=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:300],
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        pcm_file = Path(pcm_path)
        if not pcm_file.is_file() or pcm_file.stat().st_size == 0:
            logger.error("ffmpeg produced empty PCM output")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        # Step 2: PCM → SILK via pilk
        pilk.encode(pcm_path, silk_path, pcm_rate=sample_rate)

        silk_file = Path(silk_path)
        if not silk_file.is_file() or silk_file.stat().st_size == 0:
            logger.error("pilk produced empty SILK output")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        return silk_path

    except Exception as exc:
        logger.error("ogg_to_silk failed: %s", exc, exc_info=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def get_audio_duration_s(audio_path: str) -> int:
    """Return the duration of an audio file in whole seconds.

    Supports .silk (via pilk), .ogg/.opus (via ffprobe), and falls back
    to 0 on any error.
    """
    path = Path(audio_path)
    suffix = path.suffix.lower()

    # SILK — use pilk.get_duration (returns ms)
    if suffix == ".silk" and _has_pilk():
        try:
            ms = pilk.get_duration(str(path))
            return max(1, ms // 1000)
        except Exception:
            pass

    # OGG/Opus/WAV/MP3 — try ffprobe
    ffprobe = shutil.which("ffprobe")
    if ffprobe and path.is_file():
        try:
            import subprocess

            cmd = [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return max(1, int(float(result.stdout.strip())))
        except Exception:
            pass

    return 0


def cleanup_silk_dir(silk_path: Optional[str]) -> None:
    """Remove the temporary directory containing a converted SILK file."""
    if silk_path:
        tmp_dir = os.path.dirname(silk_path)
        if tmp_dir and tmp_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(tmp_dir, ignore_errors=True)
