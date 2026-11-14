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


@pytest.mark.platforms("linux")
def test_watcher_judges_the_old_gateway_by_pid_and_start_time(tmp_path, monkeypatch):
    """A PID reused by an unrelated process ends the wait (no deadline needed); the old gateway's
    own (pid, start time) keeps it waiting. A reused PID is simulated by arming the watcher with a
    start time the live process does not have."""
    from hermes_cli import _subprocess_compat

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    real_start = _subprocess_compat.process_start_time_stdlib(old.pid)
    spawned = []
    real_popen = subprocess.Popen

    def _capture(argv, **kwargs):
        proc = real_popen(argv, **kwargs)
        spawned.append(proc)
        return proc

    def _arm(marker, start_time):
        monkeypatch.setattr(_subprocess_compat, "process_start_time_stdlib", lambda pid: start_time)
        relaunch = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
        with monkeypatch.context() as m:
            m.setattr(subprocess, "Popen", _capture)
            assert gateway._spawn_gateway_restart_watcher(old.pid, relaunch, host=False)
        return spawned[-1]

    try:
        same, reused = tmp_path / "same", tmp_path / "reused"
        waiting = _arm(same, real_start)
        _arm(reused, real_start + 1)
        wait_until(reused.exists, timeout=30, what="respawn once the pid is no longer the old gateway")
        assert old.poll() is None
        started = time.monotonic()
        wait_until(lambda: same.exists() or time.monotonic() - started >= 2.0, timeout=30,
                   what="watcher with the right identity respawned or kept waiting")
        assert not same.exists() and waiting.poll() is None, "the watcher stopped waiting on the live old gateway"
    finally:
        old.kill()
        old.wait(timeout=30)
        for proc in spawned:
            proc.wait(timeout=30)
