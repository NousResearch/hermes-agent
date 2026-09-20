"""Timeout must never silently lose a bot-chat alert (2026-09-19 docgen-deadman RCA).

Covers three behaviors:
1. A timed-out CLI delivery queues a SHORT degraded-delivery marker via the deferred
   lane (references the saved output; never repeats the full payload).
2. A marker that itself times out is never re-marked (recursion guard).
3. A turn report appearing in the kill window books the delivery instead of raising
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


def _capturing_defer():
    captured = {}
    mock = Mock(side_effect=lambda key, job, content, profile, home, **kw:
                captured.update(key=key, content=content, profile=profile,
                                home=home, kw=kw) or {"id": key, "status": "queued"})
    return mock, captured


def test_timeout_queues_degraded_marker(cli_lane, monkeypatch):
    from cron import bot_chat_delivery as queue
    defer_mock, captured = _capturing_defer()
    monkeypatch.setattr(queue, "defer", defer_mock)

    job = {"id": "job-1", "name": "docgen deadman", "execution_id": "exec-1"}
    result = delivery._deliver_to_bot_chat(job, "P1 findings: everything on fire", "")

    assert defer_mock.call_count == 1
    assert captured["key"].endswith("-degraded")
    marker = captured["content"]
    assert "DELIVERY DEGRADED" in marker
    assert "docgen deadman" in marker
    assert "job-1" in marker
    assert "hermes cron runs" in marker
    assert "P1 findings: everything on fire" in marker  # short excerpt
    assert result is not None and "degraded-delivery notice was queued" in result
    assert "timed out" in result


def test_timeout_marker_not_recursive(cli_lane, monkeypatch):
    """A marker's own timeout must not queue another marker."""
    from cron import bot_chat_delivery as queue
    defer_mock, _ = _capturing_defer()
    monkeypatch.setattr(queue, "defer", defer_mock)

    marker_payload = ("[Cronjob \"x\" — DELIVERY DEGRADED, scheduled job, not the user. "
                      "Excerpt: boom]")
    job = {"id": "job-1", "execution_id": "exec-1"}
    result = delivery._deliver_to_bot_chat(job, marker_payload, "")

    defer_mock.assert_not_called()
    assert result is not None and "the result is saved" in result
    assert "degraded-delivery notice was queued" not in result


def test_timeout_defer_failure_falls_back(cli_lane, monkeypatch):
    """A broken deferred lane degrades to the old message, never raises."""
    from cron import bot_chat_delivery as queue
    monkeypatch.setattr(queue, "defer", Mock(side_effect=OSError("disk full")))

    job = {"id": "job-1", "execution_id": "exec-1"}
    result = delivery._deliver_to_bot_chat(job, "alert body", "")

    assert result is not None and "the result is saved" in result
    assert "timed out" in result


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

    def fake_read(path, pid):
        # Only after the kill: simulate the report landing in the kill window.
        if pid == 4242 and _late_state.get("killed"):
            return {"pid": pid, "exit_code": 0, "error": None}
        return None

    late_state = _late_state = {}
    real_kill = _FakeProc.kill

    def kill_and_flag(self):
        real_kill(self)
        late_state["killed"] = True

    monkeypatch.setattr(_FakeProc, "kill", kill_and_flag)
    monkeypatch.setattr(qsq, "read_turn_report", fake_read)

    result = delivery._run_bot_chat_turn(["hermes"], {}, str(tmp_path / "r.json"), 0.4)
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0


def test_no_report_still_raises_timeout(tmp_path, monkeypatch):
    """Polarity: with no report at all the timeout still raises (lost turn detected)."""
    from hermes_cli import quiet_single_query as qsq
    monkeypatch.setattr(delivery.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(qsq, "read_turn_report", lambda path, pid: None)

    with pytest.raises(subprocess.TimeoutExpired):
        delivery._run_bot_chat_turn(["hermes"], {}, str(tmp_path / "r.json"), 0.4)
