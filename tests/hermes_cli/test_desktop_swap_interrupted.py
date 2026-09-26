"""A desktop swap after an interrupted one never leaves ``release/`` without an app.

The stage-and-swap moves the live app to ``<unpacked>.previous`` and then the staged app to the
live name. A swap killed between those two renames leaves ``.previous`` as the only app on disk.
The next swap must treat it as the live app: removing it first, and then failing to move the staged
app in, would leave no app at all.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

from hermes_cli import main_desktop


def _packaged_exe_rel() -> Path:
    if sys.platform == "darwin":
        return Path("mac-arm64") / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    if sys.platform == "win32":
        return Path("win-unpacked") / "Hermes.exe"
    return Path("linux-unpacked") / "hermes"


def test_swap_after_an_interrupted_swap_keeps_the_only_app_when_it_fails(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    release = desktop_dir / "release"
    unpacked, *inside = _packaged_exe_rel().parts
    live_exe = release / unpacked / Path(*inside)
    # The interrupted swap's state: the old app aside, nothing at the live name.
    aside_exe = release / (unpacked + main_desktop._DESKTOP_PREVIOUS_SUFFIX) / Path(*inside)
    aside_exe.parent.mkdir(parents=True)
    aside_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")
    monkeypatch.setattr(main_desktop, "_stop_desktop_processes_locking_build", lambda d, **kw: [])
    real_rename = os.rename

    def disk_full_on_promotion(src, dst):
        if Path(src) == staging / unpacked:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_rename(src, dst)

    monkeypatch.setattr(main_desktop.os, "rename", disk_full_on_promotion)

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8-sig") == "old"
