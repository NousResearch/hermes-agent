"""The detached update restart watcher must not duplicate a still-running gateway."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import gateway
from gateway import status


@pytest.mark.parametrize("old_pid_alive,respawns", [(True, False), (False, True)])
def test_detached_restart_watcher_requires_old_pid_to_exit(monkeypatch, tmp_path, old_pid_alive, respawns):
    calls = []

    def capture_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", capture_popen)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: old_pid_alive)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert gateway._spawn_gateway_restart_watcher(12345, [sys.executable, "-c", "pass"])
    watcher = calls.pop()[0][2]
    monkeypatch.setattr(sys, "argv", [sys.executable, "12345", sys.executable, "-c", "pass"])
    if old_pid_alive:
        # Expire the deadline without waiting two minutes in a real detached process.
        import time
        start = time.monotonic()
        # The first clock read establishes the deadline; the second must expire it.
        ticks = iter((start, start + 121))
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
        with pytest.raises(SystemExit) as exit_info:
            exec(watcher, {"__name__": "__main__"})
        assert exit_info.value.code == 0
    else:
        exec(watcher, {"__name__": "__main__"})
    assert bool(calls) is respawns
    if respawns:
        assert calls[0][0] == [sys.executable, "-c", "pass"]
