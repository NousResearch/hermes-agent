"""``hermes update --force`` stops the update's gateways at once instead of draining them."""

import json
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


def test_force_kills_the_unit_when_its_stop_job_outlives_the_gateway(monkeypatch, tmp_path, no_record):
    """The gateway exits but its unit's stop job hangs on another process: the force deadline
    still kills the unit and waits for the job, instead of reporting failure (and rolling back)."""
    import threading
    released, kills = threading.Event(), []
    monkeypatch.setattr(force_stop, "FORCE_KILL_AFTER_S", 0.5)
    monkeypatch.setattr(force_stop, "_live_work", lambda entry: [])
    monkeypatch.setattr(posix_pause, "_stop_at_once", lambda entry: released.wait(10))
    monkeypatch.setattr(force_stop, "_kill", lambda entry: (kills.append(entry["unit"]), released.set()))
    gone = {"pid": 2 ** 22 + 7, "home": str(tmp_path), "ct": None, "kind": "systemd", "scope": "user",
            "unit": "hermes-gateway.service"}
    force_stop.force_stop_gateways({}, [gone])
    assert kills == ["hermes-gateway.service"]


def _sleeper() -> subprocess.Popen:
    """A background tool process the gateway left behind: its own session, as a scope would be."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                            stdin=subprocess.DEVNULL, start_new_session=True)


def test_force_sweeps_the_background_processes_a_killed_gateway_left_behind(monkeypatch, tmp_path, capsys,
                                                                            no_record):
    """A SIGKILLed gateway never runs ``process_registry.kill_all``: the processes its checkpoint
    names are stopped for it, and a recorded PID now held by another process is never signalled."""
    from gateway.status import get_process_start_time
    from tests.waiting import wait_until
    gateway, left, stranger = _child(ignore_term=True), _sleeper(), _sleeper()
    try:
        wait_until(lambda: get_process_start_time(left.pid) and get_process_start_time(stranger.pid))
        (tmp_path / "processes.json").write_text(json.dumps([
            {"session_id": "proc_left", "pid": left.pid, "pid_scope": "host", "systemd_unit": "",
             "host_start_time": get_process_start_time(left.pid), "persist_on_release": True,
             "command": "sleep 900"},
            {"session_id": "proc_recycled", "pid": stranger.pid, "pid_scope": "host", "systemd_unit": "",
             "host_start_time": get_process_start_time(stranger.pid) + 1, "command": "sleep 900"}]))
        monkeypatch.setattr(force_stop, "FORCE_KILL_AFTER_S", 0.5)
        monkeypatch.setattr(force_stop, "_live_work", lambda entry: [])
        force_stop.force_stop_gateways({}, [{"pid": gateway.pid, "home": str(tmp_path), "ct": None}])
        assert gateway.wait(timeout=5) == -signal.SIGKILL
        wait_until(lambda: left.poll() is not None, timeout=10, what="the left-behind process to be stopped")
        assert stranger.poll() is None, "a recycled PID (start time mismatch) was signalled"
        assert f"gateway PID {gateway.pid}: also stopped 1 background process(es) it left behind" \
            in capsys.readouterr().out
    finally:
        for proc in (gateway, left, stranger):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def test_an_unsettled_force_stop_aborts_before_the_checkout_moves(monkeypatch, tmp_path, no_record):
    """The unit's stop job outlives the force deadline and its kill: nothing may be started behind
    it, so the commit point refuses (no swap), nothing is resumed now, and the record stays owed."""
    import threading
    from hermes_cli import update_pause_record as pause_record
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    released = threading.Event()
    unit = {"kind": "systemd", "scope": "user", "unit": "hermes-gateway.service", "pid": 2 ** 22 + 9,
            "home": str(tmp_path), "cgroup": None}
    monkeypatch.setattr(posix_pause, "_discover_systemd", lambda: [dict(unit)])
    monkeypatch.setattr(posix_pause, "_discover_launchd", list)
    monkeypatch.setattr(posix_pause, "_left_running", lambda units, jobs: (units, jobs, []))
    monkeypatch.setattr(posix_pause, "_discover_bare", lambda service_pids: ([], []))
    monkeypatch.setattr(posix_pause, "_stop_at_once", lambda entry: released.wait(30))
    monkeypatch.setattr(posix_pause, "resume_paused_set",
                        lambda token: pytest.fail("a start was queued behind a stop still running"))
    monkeypatch.setattr(force_stop, "_kill", lambda entry: None)
    monkeypatch.setattr(force_stop, "_live_work", lambda entry: [])
    monkeypatch.setattr(force_stop, "FORCE_KILL_AFTER_S", 0.2)
    monkeypatch.setattr(force_stop, "_STOP_JOB_SETTLE_S", 0.2)
    token = {**posix_pause._empty_token(), "force": True}
    monkeypatch.setattr(posix_pause, "_RUN", token)
    try:
        refused = posix_pause.pause_at_commit_point()
    finally:
        released.set()
    assert refused and "hermes-gateway.service" in refused, f"no refusal naming the unit: {refused!r}"
    assert not token["resume_needed"], "this process would start the unit behind its stop job at exit"
    owed = pause_record.read()
    assert owed and owed["token"]["resume_needed"], "the stopped unit is no longer owed a restart"
    assert [u["unit"] for u in owed["token"]["posix_units"]] == ["hermes-gateway.service"]
