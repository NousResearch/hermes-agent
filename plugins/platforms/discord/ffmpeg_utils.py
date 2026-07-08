"""Shared ffmpeg executable discovery for Discord voice paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_ffmpeg_executable() -> str:
    """Return an ffmpeg command that also covers common Windows installs."""
    explicit = os.getenv("FFMPEG_PATH")
    if explicit and explicit.strip():
        return os.path.expandvars(os.path.expanduser(explicit.strip()))

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        packages_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        candidates = sorted(packages_dir.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"))
        if candidates:
            return str(candidates[-1])

    return "ffmpeg"
