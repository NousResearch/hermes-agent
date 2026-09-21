"""Desktop stage-and-swap promotion rides out a transient Windows file lock (#112544).

A real-time scanner (AV/EDR) briefly holds ``release/win-unpacked`` right after the pack; the
promotion rename then raises PermissionError and the updater used to report "Could not install the
rebuilt desktop app" even though a retry a second later would have succeeded.
"""

from __future__ import annotations

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


def _staged_over_live(tmp_path: Path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")
    # The swap point now passes ``also_posix=True`` (#116504); the double must accept the keyword.
    monkeypatch.setattr(main_desktop, "_stop_desktop_processes_locking_build", lambda d, **kw: [])
    slept: list[float] = []
    monkeypatch.setattr(main_desktop._time_mod, "sleep", slept.append)
    return desktop_dir, staging, live_exe, slept


def test_swap_retries_transient_permission_error_then_promotes(tmp_path, monkeypatch, caplog):
    desktop_dir, staging, live_exe, slept = _staged_over_live(tmp_path, monkeypatch)
    real_rename = os.rename
    locked = {"n": 0}

    def scanner_locked_rename(src, dst):
        if Path(dst) == live_exe.parent and locked["n"] < 2:
            locked["n"] += 1
            raise PermissionError(32, "being used by another process")
        return real_rename(src, dst)

    monkeypatch.setattr(main_desktop.os, "rename", scanner_locked_rename)
    with caplog.at_level("WARNING", logger=main_desktop.logger.name):
        promoted = main_desktop._swap_staged_desktop_app(desktop_dir, staging)

    assert promoted == live_exe
    assert live_exe.read_text(encoding="utf-8") == "new"
    assert slept == [0.5, 1.0]
    assert sum("hit a file lock" in r.message for r in caplog.records) == 2
    assert not staging.exists()


def test_swap_gives_up_after_bounded_retries_and_keeps_live_app(tmp_path, monkeypatch, caplog):
    """A lock that never clears: bounded attempts, real OSError surfaced in the log, live app untouched."""
    desktop_dir, staging, live_exe, slept = _staged_over_live(tmp_path, monkeypatch)
    real_rename = os.rename
    attempts = {"n": 0}

    def always_locked_rename(src, dst):
        if Path(src) == staging / _packaged_exe_rel().parts[0]:
            attempts["n"] += 1
            raise PermissionError(5, "Access is denied")
        return real_rename(src, dst)

    monkeypatch.setattr(main_desktop.os, "rename", always_locked_rename)
    with caplog.at_level("WARNING", logger=main_desktop.logger.name):
        assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None

    assert attempts["n"] == len(main_desktop._DESKTOP_SWAP_RENAME_RETRY_DELAYS_S) + 1
    assert len(slept) == len(main_desktop._DESKTOP_SWAP_RENAME_RETRY_DELAYS_S)
    assert live_exe.read_text(encoding="utf-8") == "old"
    assert not (live_exe.parent.parent / (live_exe.parent.name + ".previous")).exists()
    assert any("Access is denied" in r.message and "live app kept" in r.message for r in caplog.records)
