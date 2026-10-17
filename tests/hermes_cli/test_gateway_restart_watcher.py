"""keep the detached update watcher from restarting against a live process"""

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_cli import _subprocess_compat, gateway
from gateway import status


@pytest.mark.parametrize(
    ("pid_alive", "expected_start_time", "current_start_time", "respawns"),
    [
        (True, 100, 100, False),
        (True, 100, 199, False),
        (True, 100, 100 + status.START_TIME_DRIFT_TOLERANCE, False),
        (True, 100, 101 + status.START_TIME_DRIFT_TOLERANCE, True),
        (False, 100, 100, True),
        (True, 100, 400, True),
        (True, None, 100, False),
        (False, None, 100, True),
        (True, 100, None, False),
        (True, OSError("cannot read parent fingerprint"), 100, False),
        (False, OSError("cannot read parent fingerprint"), 100, True),
        (True, 100, OSError("cannot read child fingerprint"), False),
    ],
)
def test_detached_watcher_respects_process_identity(
    monkeypatch, tmp_path, pid_alive, expected_start_time, current_start_time, respawns
):
    calls = []

    def capture_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(subprocess, "Popen", capture_popen)
    readings = iter((expected_start_time, current_start_time))

    def read_start_time(pid):
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(_subprocess_compat, "pid_exists_stdlib", lambda pid: pid_alive)
    monkeypatch.setattr(status, "get_process_start_time", read_start_time)
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


@pytest.mark.parametrize("pid_alive", [True, False])
@pytest.mark.parametrize("expected_start_time", [100, None, OSError("unreadable fingerprint")])
def test_bare_watcher_only_restarts_a_dead_pid(monkeypatch, tmp_path, pid_alive, expected_start_time):
    """Real generated code must fail safe without site-packages, even after a failed parent probe."""
    if pid_alive:
        pid = os.getpid()
    else:
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=10)
        pid = proc.pid

    def read_start_time(pid):
        if isinstance(expected_start_time, Exception):
            raise expected_start_time
        return expected_start_time

    marker = tmp_path / "respawned"
    command = [sys.executable, "-I", "-S", "-c", f"open({str(marker)!r}, 'w').write('ok')"]
    calls = []
    monkeypatch.setattr(status, "get_process_start_time", read_start_time)
    monkeypatch.setattr(gateway, "GATEWAY_RESTART_WATCHER_TIMEOUT_S", 0)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    with monkeypatch.context() as capture:
        capture.setattr(subprocess, "Popen", lambda argv, **kwargs: calls.append(argv))
        assert gateway._spawn_gateway_restart_watcher(pid, command, host=False)

    # -I ignores PYTHONPATH and -S hides dependency site-packages; cwd is also outside the checkout.
    watcher = calls[0]
    launch_marker = tmp_path / "launch-requested"
    audit = (
        "import sys\n"
        "def record_spawn(event, args):\n"
        "    if event == 'subprocess.Popen':\n"
        f"        open({str(launch_marker)!r}, 'w').write('spawn')\n"
        "sys.addaudithook(record_spawn)\n"
    )
    result = subprocess.run(
        [watcher[0], "-I", "-S", "-c", audit + watcher[2], *watcher[3:]], cwd=tmp_path, capture_output=True, text=True,
        encoding="utf-8", timeout=10,
    )
    assert result.returncode == 0, result.stderr
    # The audit event is synchronous: a completed watcher cannot hide a late child launch.
    assert launch_marker.exists() is not pid_alive
    if pid_alive:
        assert not marker.exists()
    else:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if marker.exists() and marker.read_text() == "ok":
                break
            time.sleep(0.05)
        assert marker.read_text() == "ok"
