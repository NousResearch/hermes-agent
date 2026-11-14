"""``hermes update --force`` stops the update's gateways at once instead of draining them."""

import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import update_cmd_force_stop as force_stop
from hermes_cli import update_cmd_posix_pause as posix_pause

pytestmark = pytest.mark.platforms("posix")


def _child(ignore_term: bool) -> subprocess.Popen:
    code = ("import signal, time\n"
            + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
            + "print('ready', flush=True)\n"
            + "time.sleep(600)\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"  # the SIGTERM disposition is set
    return proc


@pytest.fixture
def no_record(monkeypatch):
    from hermes_cli import update_pause_record
    monkeypatch.setattr(update_pause_record, "mark_stop_sent", lambda token, pid: None)


def test_force_pause_never_asks_for_a_drain_and_stops_within_the_bound(monkeypatch, tmp_path, no_record):
    """A live gateway that ignores SIGTERM (a turn the drain would wait 30 min on) is gone within
    the force bound, and the drain request is never sent."""
    proc = _child(ignore_term=True)
    try:
        monkeypatch.setattr(force_stop, "FORCE_KILL_AFTER_S", 1.0)
        monkeypatch.setattr(posix_pause, "_ask_to_drain",
                            lambda entry: pytest.fail("--force must not request a drain"))
        monkeypatch.setattr(posix_pause, "_drain_budget", lambda: 1800.0)
        entry = {"pid": proc.pid, "home": str(tmp_path), "argv": ["x"], "ct": None}
        started = time.monotonic()
        posix_pause._stop_gateways({"force": True}, [entry])
        assert time.monotonic() - started < 10.0
        assert proc.wait(timeout=5) == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()


def test_force_stop_names_what_it_interrupted(monkeypatch, tmp_path, capsys, no_record):
    proc = _child(ignore_term=False)
    try:
        monkeypatch.setattr(force_stop, "_live_work", lambda entry: [
            {"kind": "chat", "session": "agent:main:api_server:dm:s1", "current_tool": "terminal"},
            {"kind": "delegation", "delegation_id": "deleg_1", "goal": "child"}])
        force_stop.force_stop_gateways({}, [{"pid": proc.pid, "home": str(tmp_path), "ct": None}])
        assert proc.wait(timeout=5) == -signal.SIGTERM
        out = capsys.readouterr().out
        assert f"gateway PID {proc.pid}: stopped by --force, interrupting 2 unit(s)" in out
        assert "background delegation deleg_1" in out
    finally:
        if proc.poll() is None:
            proc.kill()
