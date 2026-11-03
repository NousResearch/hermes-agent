"""Read-only discovery of native Chromium browsers on Windows ARM64."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Iterable

from hermes_platform.host import facts


def _pe_machine(path: str) -> int | None:
    """Read the executable's architecture; folder names do not distinguish ARM64 from emulated x64."""
    try:
        with open(path, "rb") as binary:
            if binary.read(2) != b"MZ":
                return None
            binary.seek(0x3C)
            offset = struct.unpack("<I", binary.read(4))[0]
            binary.seek(offset)
            if binary.read(4) != b"PE\0\0":
                return None
            return struct.unpack("<H", binary.read(2))[0]
    except (OSError, struct.error):
        return None


def _registered_windows_browsers() -> Iterable[str]:
    """Stable Edge, then Chrome, in per-user and machine App Paths (both registry views)."""
    import winreg

    for name in ("msedge.exe", "chrome.exe"):
        key = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{name}"
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                try:
                    with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view) as entry:
                        path, _ = winreg.QueryValueEx(entry, "")
                except OSError:
                    continue
                if isinstance(path, str) and path.strip():
                    yield os.path.expandvars(path.strip().strip('"'))


def _first_arm64_browser(candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if _pe_machine(candidate) == 0xAA64:
            return str(Path(candidate))
    return None


def native_windows_chromium() -> str | None:
    """An installed native ARM64 Edge/Chrome binary, without launching it or reading its user profile."""
    if facts.os_family() != "win32" or facts.native_arch() != "arm64":
        return None
    return _first_arm64_browser(_registered_windows_browsers())
