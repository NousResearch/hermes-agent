"""Timeout must never silently lose a bot-chat alert (2026-09-19 docgen-deadman RCA).

Two invariants:
1. A timed-out CLI delivery queues exactly one SHORT degraded-delivery marker via the
   deferred lane (flagged on the record, references `hermes cron runs`, never repeats the
   full payload); a timeout while draining that marker queues nothing further.
2. A turn report appearing in the kill window books the delivery instead of raising
   TimeoutExpired (a delivered turn must not be reported as lost).
"""
import subprocess
import threading
from unittest.mock import Mock

import pytest

from cron import scheduler_delivery as delivery


@pytest.fixture()
def cli_lane(tmp_path, monkeypatch):
    """Route _deliver_to_bot_chat straight into the CLI fallback lane."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        delivery, "_run_bot_chat_turn",
        Mock(side_effect=subprocess.TimeoutExpired(["hermes"], 600)))
    return tmp_path


def test_timeout_queues_degraded_marker(cli_lane, monkeypatch):
    from cron import bot_chat_delivery as queue
    calls = []
    monkeypatch.setattr(queue, "defer", Mock(
        side_effect=lambda key, job, content, profile, home, **kw:
        calls.append(dict(key=key, content=content, kw=kw)) or {"id": key, "status": "queued"}))

    job = {"id": "job-1", "name": "docgen deadman", "execution_id": "exec-1"}
    payload = "P1 findings: everything on fire\n" + "detail line\n" * 40
    result = delivery._deliver_to_bot_chat(job, payload, "")

    assert len(calls) == 1
    key, marker, kw = calls[0]["key"], calls[0]["content"], calls[0]["kw"]
    assert len(key) == 64 and int(key, 16) >= 0  # a fresh hex delivery id, not "<key>-degraded"
    assert kw.get("degraded") is True
    assert "DELIVERY DEGRADED" in marker and "job-1" in marker
    assert not marker.startswith("[")  # a plain body: the standard cronjob header wraps it
    assert "hermes cron runs" in marker
    assert "P1 findings: everything on fire" in marker  # short excerpt only...
    assert payload.strip() not in marker  # ...never the full payload
    assert result is not None and "degraded-delivery notice was queued" in result

    # Draining the marker record itself times out too: recognised by the record's flag,
    # so nothing further is queued (the guard is structural, not a text match).
    from hermes_state import SessionDB
    SessionDB(db_path=cli_lane / "state.db").close()  # a deferred target must have a state.db
    record = {"id": key, "home": str(cli_lane), "profile": "", "degraded": True}
    posted = []

    def capture_then_timeout(argv, env, report_file, timeout_s):
        with open(argv[argv.index("--query-file") + 1], encoding="utf-8") as fh:
            posted.append(fh.read())
        raise subprocess.TimeoutExpired(argv, timeout_s)

    monkeypatch.setattr(delivery, "_run_bot_chat_turn", capture_then_timeout)
    result = delivery._deliver_to_bot_chat(job, marker, "", deferred=record)
    assert len(calls) == 1
    assert result is not None and "the result is saved" in result
    assert "degraded-delivery notice was queued" not in result
    # The drained marker is delivered like any other output: one standard header (naming
    # the job) and the plain marker body — no second, marker-specific framing.
    assert len(posted) == 1
    assert posted[0].startswith('[Cronjob "docgen deadman" output — ')
    assert posted[0].count("[Cronjob") == 1
    assert posted[0].endswith("\n\n" + marker)


class _FakeProc:
    """Blocks in communicate() until killed; exit code set by the report path."""

    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self.returncode = None
        self.killed = threading.Event()

    def communicate(self):
        self.killed.wait()
        return "", ""

    def kill(self):
        self.returncode = -9
        self.killed.set()


def test_late_turn_report_books_delivery(tmp_path, monkeypatch):
    """Report appearing in the kill window = turn completed = booked, not a timeout."""
    from hermes_cli import quiet_single_query as qsq
    monkeypatch.setattr(delivery.subprocess, "Popen", _FakeProc)
    late_state = {}

    def fake_read(path, pid):
        # Only after the kill: simulate the report landing in the kill window.
        if pid == 4242 and late_state.get("killed"):
            return {"pid": pid, "exit_code": 0, "error": None}
        return None

    real_kill = _FakeProc.kill

    def kill_and_flag(self):
        real_kill(self)
        late_state["killed"] = True

    monkeypatch.setattr(_FakeProc, "kill", kill_and_flag)
    monkeypatch.setattr(qsq, "read_turn_report", fake_read)

    result = delivery._run_bot_chat_turn(["hermes"], {}, str(tmp_path / "r.json"), 0.4)
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0
