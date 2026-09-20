"""The desktop promotion sweep stops cwd-based directory locks, not just exe locks.

A process whose EXE lives outside ``release`` but whose WORKING DIRECTORY sits inside
``release/win-unpacked`` blocks the promotion rename (``win-unpacked -> win-unpacked.previous``)
with WinError 32 even with zero Hermes.exe processes alive: Windows refuses to rename a directory
that is any process's cwd. Observed with SogouCloud.exe (input-method cloud component, exe under
``Program Files (x86)\\SogouInput``) inheriting the packaged app's cwd, silently keeping four
successful rebuilds from installing. ``_stop_desktop_processes_locking_build`` only matched the
exe path, so the sweep found nothing to stop and every promotion failed.

The sweep runs for real here against throwaway ``ping`` dummies spawned by the test itself —
they are the only processes whose exe/cwd sit under the fixture's ``release`` tree, so letting
the function terminate them is the honest E2E and doubles as cleanup.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

import psutil
import pytest

from hermes_cli import main_desktop

pytestmark = pytest.mark.windows_only


def _dummy_ping(cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["cmd", "/c", "ping", "-n", "30", "127.0.0.1"],
        cwd=str(cwd),
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _wait_gone(pid: int) -> bool:
    for _ in range(50):
        if not psutil.pid_exists(pid):
            return True
        psutil.wait_procs([psutil.Process(pid)], timeout=0.2)
    return False


def test_sweep_stops_process_whose_cwd_is_inside_release(tmp_path):
    """Regression: exe outside the tree, cwd inside it — the SogouCloud lock shape."""
    desktop_dir = tmp_path / "apps" / "desktop"
    release = desktop_dir / "release" / "win-unpacked"
    release.mkdir(parents=True)

    dummy = _dummy_ping(release)
    try:
        stopped = main_desktop._stop_desktop_processes_locking_build(desktop_dir)
        assert dummy.pid in stopped
        assert _wait_gone(dummy.pid), "sweep claimed the pid but the process survived"
    finally:
        with contextlib.suppress(Exception):
            dummy.kill()
            dummy.wait(timeout=10)


def test_sweep_ignores_process_with_cwd_outside_release(tmp_path):
    """A cwd elsewhere is not a lock on the release tree; the sweep must not touch it."""
    desktop_dir = tmp_path / "apps" / "desktop"
    (desktop_dir / "release" / "win-unpacked").mkdir(parents=True)

    dummy = _dummy_ping(tmp_path)
    try:
        stopped = main_desktop._stop_desktop_processes_locking_build(desktop_dir)
        assert dummy.pid not in stopped
        assert stopped == []
        assert psutil.pid_exists(dummy.pid), "sweep killed an unrelated process"
    finally:
        dummy.kill()
        dummy.wait(timeout=10)
