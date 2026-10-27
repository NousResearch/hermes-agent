"""Absolute paths for the Windows tools that ship in the system directory.

A bare ``taskkill`` or ``powershell`` handed to ``CreateProcess`` is looked up in the parent's
current directory BEFORE System32, and Hermes's cwd is usually a repository someone else controls,
so a ``taskkill.exe`` committed to it would run instead. Spawning the absolute path never searches.
The system directory comes from ``GetSystemDirectoryW``, never from ``%SystemRoot%``.
"""

from __future__ import annotations

import functools
import ntpath
import sys

_IS_WINDOWS = sys.platform == "win32"
# Tools outside the system directory's top level, relative to it. Every other name is ``<name>.exe``.
_SUBPATHS = {
    "powershell": ("WindowsPowerShell", "v1.0", "powershell.exe"),
    "wmic": ("wbem", "WMIC.exe"),
}


@functools.cache
def _system_directory() -> str:
    import ctypes

    buf = ctypes.create_unicode_buffer(260)
    length = ctypes.WinDLL("kernel32", use_last_error=True).GetSystemDirectoryW(buf, len(buf))
    if not 0 < length < len(buf):
        raise ctypes.WinError(ctypes.get_last_error())
    return buf.value


def system_tool(name: str) -> str:
    """``name`` as the absolute path of the copy in the Windows system directory.

    Off Windows the name comes back unchanged (POSIX ``exec`` never searches the cwd). The path is
    returned whether or not the file exists: a missing tool fails the spawn with
    ``FileNotFoundError``, exactly as the bare name would, and never falls back to a search.
    """
    if not _IS_WINDOWS:
        return name
    stem = name.lower().removesuffix(".exe")
    return ntpath.join(_system_directory(), *_SUBPATHS.get(stem, (f"{stem}.exe",)))
