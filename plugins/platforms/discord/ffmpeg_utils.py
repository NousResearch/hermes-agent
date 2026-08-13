"""Shared ffmpeg executable discovery for Discord voice paths.

Discovery itself is owned by ``tools.transcription_tools`` (the same helper
the STT pipeline uses — PATH plus common Homebrew/local prefixes); this module
only layers the Discord-voice-specific extras on top: an explicit
``FFMPEG_PATH`` override and a Windows winget fallback for installs that
never touch PATH.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _shared_find_ffmpeg():
    """Delegate to the repo-wide ffmpeg discovery helper when importable."""
    try:
        from tools.transcription_tools import _find_ffmpeg_binary
    except ImportError:  # standalone plugin import (tests / sandboxes)
        return shutil.which("ffmpeg")
    return _find_ffmpeg_binary()


def resolve_ffmpeg_executable() -> str:
    """Return an ffmpeg command that also covers common Windows installs."""
    explicit = os.getenv("FFMPEG_PATH")
    if explicit and explicit.strip():
        return os.path.expandvars(os.path.expanduser(explicit.strip()))

    discovered = _shared_find_ffmpeg()
    if discovered:
        return discovered

    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        packages_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        candidates = sorted(packages_dir.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"))
        if candidates:
            return str(candidates[-1])

    return "ffmpeg"
