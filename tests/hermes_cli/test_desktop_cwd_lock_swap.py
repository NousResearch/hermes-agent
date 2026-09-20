"""CWD-lock detection + sharing-violation retry for desktop stage-and-swap (#107541).

A third-party helper whose exe lives outside ``release/`` but whose CWD is
inside ``release/win-unpacked`` holds a Windows directory handle that blocks
``os.rename(live_root, previous)`` with WinError 32. These tests pin the
Windows-only CWD second pass, fail-open controls, rename retry scope, and
the user-facing OSError on promote failure.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop


_SOGOU_EXE = r"C:\Program Files (x86)\SogouInput\SOGOUSmartAssistant.exe"


def _sharing(winerror=32):
    e = OSError(winerror, "The process cannot access the file")
    e.winerror = winerror
    return e


def _packaged_exe_rel() -> Path:
    if sys.platform == "darwin":
        return Path("mac-arm64") / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    if sys.platform == "win32":
        return Path("win-unpacked") / "Hermes.exe"
    return Path("linux-unpacked") / "hermes"


class FakeProc:
    def __init__(self, pid, exe, cwd=None, cwd_error=None):
        self.pid = pid
        self.info = {"pid": pid, "exe": exe}
        self._cwd = cwd
        self._cwd_error = cwd_error
        self.terminated = False
        self.killed = False

    def cwd(self):
        if self._cwd_error is not None:
            raise self._cwd_error
        if not self._cwd:
            raise FileNotFoundError("cwd")
        return self._cwd

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _install_psutil(monkeypatch, procs):
    fake = types.SimpleNamespace(
        process_iter=lambda attrs=None: iter(procs),
        wait_procs=lambda procs, timeout=5: ([], []),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def _win32_desktop(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main_desktop.sys, "platform", "win32")
    desktop_dir = tmp_path / "apps" / "desktop"
    release = desktop_dir / "release"
    (release / "win-unpacked").mkdir(parents=True)
    return desktop_dir, release


def _swap_trees(tmp_path: Path):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    live_exe = desktop_dir / "release" / _packaged_exe_rel()
    live_exe.parent.mkdir(parents=True)
    live_exe.write_text("old", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    staged_exe = staging / _packaged_exe_rel()
    staged_exe.parent.mkdir(parents=True)
    staged_exe.write_text("new", encoding="utf-8")
    return desktop_dir, staging, live_exe


# ─── Detection ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cwd_under",
    ["win-unpacked", ""],
    ids=["cwd_in_win_unpacked", "cwd_is_release_dir"],
)
def test_win32_cwd_inside_release_selects_third_party_helper(tmp_path, monkeypatch, cwd_under):
    desktop_dir, release = _win32_desktop(tmp_path, monkeypatch)
    cwd = str((release / cwd_under).resolve()) if cwd_under else str(release.resolve())
    proc = FakeProc(20776, _SOGOU_EXE, cwd=cwd)
    _install_psutil(monkeypatch, [proc])

    stopped = main_desktop._stop_desktop_processes_locking_build(desktop_dir)

    assert 20776 in stopped
    assert proc.terminated is True


def test_win32_missing_exe_still_scans_cwd(tmp_path, monkeypatch):
    desktop_dir, release = _win32_desktop(tmp_path, monkeypatch)
    proc = FakeProc(20776, None, cwd=str((release / "win-unpacked").resolve()))
    _install_psutil(monkeypatch, [proc])

    stopped = main_desktop._stop_desktop_processes_locking_build(desktop_dir)

    assert 20776 in stopped
    assert proc.terminated is True


# ─── Fail-open controls ─────────────────────────────────────────────────────


def test_cwd_outside_release_not_selected(tmp_path, monkeypatch):
    desktop_dir, _release = _win32_desktop(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = FakeProc(20776, _SOGOU_EXE, cwd=str(elsewhere.resolve()))
    _install_psutil(monkeypatch, [proc])

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == []
    assert proc.terminated is False


def test_ancestor_cwd_desktop_dir_not_selected(tmp_path, monkeypatch):
    desktop_dir, _release = _win32_desktop(tmp_path, monkeypatch)
    proc = FakeProc(20776, _SOGOU_EXE, cwd=str(desktop_dir.resolve()))
    _install_psutil(monkeypatch, [proc])

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == []
    assert proc.terminated is False


def test_cwd_access_denied_skipped(tmp_path, monkeypatch):
    desktop_dir, _release = _win32_desktop(tmp_path, monkeypatch)
    proc = FakeProc(20776, _SOGOU_EXE, cwd_error=Exception("AccessDenied"))
    _install_psutil(monkeypatch, [proc])

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == []
    assert proc.terminated is False


def test_non_win32_returns_empty_even_if_cwd_locks_release(tmp_path, monkeypatch):
    monkeypatch.setattr(main_desktop.sys, "platform", "darwin")
    desktop_dir = tmp_path / "apps" / "desktop"
    release = desktop_dir / "release"
    (release / "win-unpacked").mkdir(parents=True)
    proc = FakeProc(20776, _SOGOU_EXE, cwd=str((release / "win-unpacked").resolve()))
    _install_psutil(monkeypatch, [proc])

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == []
    assert proc.terminated is False


def test_exe_under_release_still_selected(tmp_path, monkeypatch):
    desktop_dir, release = _win32_desktop(tmp_path, monkeypatch)
    hermes_exe = release / "win-unpacked" / "Hermes.exe"
    hermes_exe.write_text("", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = FakeProc(4242, str(hermes_exe.resolve()), cwd=str(elsewhere.resolve()))
    _install_psutil(monkeypatch, [proc])

    stopped = main_desktop._stop_desktop_processes_locking_build(desktop_dir)

    assert 4242 in stopped
    assert proc.terminated is True


# ─── Sharing-violation retry / EXDEV ────────────────────────────────────────


def test_sharing_violation_on_live_rename_is_retried(tmp_path, monkeypatch):
    desktop_dir, staging, live_exe = _swap_trees(tmp_path)
    monkeypatch.setattr(main_desktop._time_mod, "sleep", lambda _s: None)
    real_rename = cli_main.os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _sharing(32)
        return real_rename(src, dst)

    monkeypatch.setattr(cli_main.os, "rename", flaky_rename)

    promoted = main_desktop._swap_staged_desktop_app(desktop_dir, staging)

    assert promoted == live_exe
    assert live_exe.read_text(encoding="utf-8") == "new"


def test_exdev_on_staged_to_live_is_not_retried(tmp_path, monkeypatch):
    desktop_dir, staging, live_exe = _swap_trees(tmp_path)
    sleeps = []
    monkeypatch.setattr(main_desktop._time_mod, "sleep", lambda s: sleeps.append(s))
    real_rename = cli_main.os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("EXDEV simulated")
        return real_rename(src, dst)

    monkeypatch.setattr(cli_main.os, "rename", flaky_rename)

    assert main_desktop._swap_staged_desktop_app(desktop_dir, staging) is None
    assert live_exe.read_text(encoding="utf-8") == "old"
    assert sleeps == []
    assert calls["n"] == 3  # live→previous, staged→live (raise), previous→live rollback


# ─── Console OSError ────────────────────────────────────────────────────────


def test_promote_surfaces_winerror32_in_console(tmp_path, monkeypatch, capsys):
    desktop_dir, staging, _live_exe = _swap_trees(tmp_path)
    monkeypatch.setattr(main_desktop, "_desktop_macos_relaunchable_fixup", lambda *a, **k: None)
    monkeypatch.setattr(main_desktop._time_mod, "sleep", lambda _s: None)

    def locked_rename(src, dst):
        raise _sharing(32)

    monkeypatch.setattr(cli_main.os, "rename", locked_rename)

    with pytest.raises(SystemExit) as excinfo:
        main_desktop._promote_staged_desktop_app(desktop_dir, staging)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Could not install the rebuilt desktop app into" in out
    assert "previous desktop app was left untouched" in out
    # Must surface the OSError itself, not only the generic install line.
    assert "The process cannot access the file" in out or "[WinError 32]" in out
