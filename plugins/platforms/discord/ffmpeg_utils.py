"""ffmpeg discovery for Discord voice: ``tools.transcription_audio`` owns the shared lookup
(PATH + Homebrew/local prefixes); this layers an explicit ``FFMPEG_PATH`` override and a
Windows winget fallback (installs that never touch PATH) on top."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _shared_find_ffmpeg():
    """Delegate to the repo-wide ffmpeg discovery helper when importable."""
    try:
        from tools.transcription_audio import _find_ffmpeg_binary
    except ImportError:  # standalone plugin import (tests / sandboxes)
        return shutil.which("ffmpeg")
    return _find_ffmpeg_binary()


def resolve_ffmpeg_executable() -> str:
    """Return an ffmpeg command that also covers common Windows installs."""
    explicit = (os.getenv("FFMPEG_PATH") or "").strip()
    if explicit:
        return os.path.expandvars(os.path.expanduser(explicit))
    if discovered := _shared_find_ffmpeg():
        return discovered
    if local_appdata := os.getenv("LOCALAPPDATA"):
        candidates = sorted((Path(local_appdata) / "Microsoft" / "WinGet" / "Packages").glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"))
        if candidates:
            return str(candidates[-1])
    return "ffmpeg"
