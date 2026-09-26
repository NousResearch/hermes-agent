"""keep the detached update watcher from restarting against a live process"""

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_cli import gateway
from gateway import status


@pytest.mark.parametrize(
    ("pid_alive", "current_start_time", "respawns"),
    [(True, 100, False), (True, 199, False), (False, 100, True), (True, 400, True)],
)
def test_detached_watcher_respects_process_identity(
    monkeypatch, tmp_path, pid_alive, current_start_time, respawns
):
    calls = []

    def capture_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", capture_popen)
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid_alive)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 100)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: current_start_time)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert gateway._spawn_gateway_restart_watcher(12345, [sys.executable, "-c", "pass"])

    watcher = calls.pop()[0][2]
    monkeypatch.setattr(sys, "argv", [sys.executable, "12345", sys.executable, "-c", "pass"])
    ticks = iter((0.0, 121.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks, 121.0))

    if not respawns:
        with pytest.raises(SystemExit) as exit_info:
            exec(watcher, {"__name__": "__main__"})
        assert exit_info.value.code == 0
    else:
        exec(watcher, {"__name__": "__main__"})

    assert bool(calls) is respawns
    if respawns:
        assert calls[0][0] == [sys.executable, "-c", "pass"]
