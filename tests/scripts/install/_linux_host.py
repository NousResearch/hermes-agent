"""Let install.sh's platform gate see a Linux host on any test host.

``check_platform`` refuses anything but Linux/macOS by ``uname -s`` (Git Bash on Windows
reports ``MINGW64_NT-…`` and is sent to install.ps1, as it should be). Tests of Linux
install behaviour that is not about the host itself put this ``uname`` first on PATH;
every query other than ``-s`` still answers from the real ``uname``.
"""
from __future__ import annotations

import os
from pathlib import Path

_UNAME = """#!/bin/sh
case "$*" in -s) echo Linux; exit 0 ;; esac
# This shim's directory is the first PATH entry; drop it to reach the real uname.
PATH="${PATH#*:}" exec uname "$@"
"""


def linux_uname_path(tmp_path: Path, path: str) -> str:
    """Return ``path`` with a directory holding a Linux-reporting ``uname`` prepended."""
    bindir = tmp_path / "linux-uname"
    bindir.mkdir()
    uname = bindir / "uname"
    uname.write_text(_UNAME, encoding="utf-8", newline="\n")
    uname.chmod(0o755)
    return str(bindir) + os.pathsep + path
