"""``hermes update`` leaves a busy manual gateway running with its restart pending, after arming the
detached restart watcher. The watcher used to respawn ``gateway run --replace`` at its 120 s
deadline, and ``--replace`` SIGTERMs/SIGKILLs the still-busy gateway: the live work the update had
just declined to stop was killed anyway. On POSIX the watcher now waits for the old PID to exit.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from hermes_cli import gateway
from tests.waiting import wait_until


@pytest.mark.platforms("posix")
def test_watcher_never_respawns_while_the_old_gateway_is_still_running(tmp_path, monkeypatch):
    release, exited, marker = tmp_path / "release", tmp_path / "old-exited", tmp_path / "respawned"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The busy gateway: runs until released, and records its own exit.
    old = subprocess.Popen([sys.executable, "-c", (
        "import os, time\n"
        f"while not os.path.exists({str(release)!r}): time.sleep(0.05)\n"
        f"open({str(exited)!r}, 'w').close()\n")])
    # The respawn records whether the old gateway was already gone when it started.
    relaunch = [sys.executable, "-c",
                f"import os; open({str(marker)!r}, 'w').write(str(os.path.exists({str(exited)!r})))"]
    # Its deadline has already passed: the old watcher respawned at once.
    monkeypatch.setattr(gateway, "GATEWAY_RESTART_WATCHER_TIMEOUT_S", 0)
    spawned = []
    real_popen = subprocess.Popen

    def _capture(argv, **kwargs):
        proc = real_popen(argv, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _capture)
    try:
        assert gateway._spawn_gateway_restart_watcher(old.pid, relaunch, host=False)
        watcher, started = spawned[0], time.monotonic()
        # Either the watcher respawned (old code), or it has been waiting well past its deadline.
        wait_until(lambda: marker.exists() or (watcher.poll() is None and time.monotonic() - started >= 2.0),
                   timeout=30, what="watcher respawned or kept waiting")
        release.touch()
        wait_until(lambda: marker.exists() and marker.read_text(), timeout=30, what="respawn after exit")
        assert marker.read_text() == "True", "the watcher respawned while the old gateway was still running"
    finally:
        release.touch()
        old.wait(timeout=30)
