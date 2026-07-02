"""Tests for gateway.runtime.run_manager.RunManager.

Covers:
- create_run lifecycle
- Event append and replay
- Status transitions
- stop_run semantics
- approval/clarify unsupported responses
- Unknown-run deterministic behaviour
- Thread safety (basic)
"""

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.models import (
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_AWAITING_APPROVAL,
    RUN_STATUS_AWAITING_CLARIFY,
    RUN_STATUS_CANCELLING,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_COMPLETED,
    EVENT_RUN_STARTED,
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
    EVENT_CLARIFY_REQUESTED,
    EVENT_CLARIFY_RESOLVED,
    EVENT_DONE,
    EVENT_ERROR,
    TERMINAL_STATUSES,
)


class TestCreateRun:
    def test_returns_run_id_session_id_status(self):
        mgr = RunManager()
        result = mgr.create_run("sess_123", message="hello")
        assert result["run_id"].startswith("run_")
        assert result["session_id"] == "sess_123"
        assert result["status"] == "queued"
        assert "controls" in result
        assert "events_url" in result
        assert "status_url" in result

    def test_appends_run_started_event(self):
        mgr = RunManager()
        result = mgr.create_run("sess_abc", message="test", model="m1/m2")
        events = mgr.read_events(result["run_id"])
        assert events is not None
        started = events["events"][0]
        assert started["type"] == EVENT_RUN_STARTED
        assert started["payload"]["message"] == "test"
        assert started["payload"]["model"] == "m1/m2"

    def test_event_id_stable_after_create(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        events = mgr.read_events(r["run_id"])
        started = events["events"][0]
        assert started["event_id"] == f"{r['run_id']}:1"
        assert started["seq"] == 1

    def test_unique_run_ids(self):
        mgr = RunManager()
        r1 = mgr.create_run("s1")
        r2 = mgr.create_run("s2")
        assert r1["run_id"] != r2["run_id"]

    def test_passes_metadata(self):
        mgr = RunManager()
        r = mgr.create_run("s", metadata={"client": "webui"})
        events = mgr.read_events(r["run_id"])
        assert events["events"][0]["payload"]["metadata"] == {"client": "webui"}

    def test_controls_on_queued_run(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        assert "observe" in r["controls"]
        assert "stop" in r["controls"]


class TestGetStatus:
    def test_returns_last_event_id_and_last_seq(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.append_event(r["run_id"], "test.event")
        status = mgr.get_status(r["run_id"])
        assert status is not None
        assert status["last_event_id"] == f"{r['run_id']}:2"
        assert status["last_seq"] == 2

    def test_unknown_run_returns_none(self):
        mgr = RunManager()
        assert mgr.get_status("run_nonexistent") is None


class TestReadEvents:
    def test_returns_events_in_order(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        mgr.append_event(r["run_id"], "event.c")
        result = mgr.read_events(r["run_id"])
        assert len(result["events"]) == 4  # run.started + 3 more
        event_types = [e["type"] for e in result["events"]]
        assert EVENT_RUN_STARTED in event_types
        assert "event.a" in event_types
        assert "event.b" in event_types
        assert "event.c" in event_types

    def test_after_seq_filters(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        mgr.append_event(r["run_id"], "event.c")
        result = mgr.read_events(r["run_id"], after_seq=1)
        event_types = [e["type"] for e in result["events"]]
        assert EVENT_RUN_STARTED not in event_types
        assert event_types == ["event.a", "event.b", "event.c"]

    def test_limit(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        result = mgr.read_events(r["run_id"], limit=2)
        assert len(result["events"]) == 2

    def test_unknown_run_returns_none(self):
        mgr = RunManager()
        assert mgr.read_events("run_nonexistent") is None

    def test_returns_run_id_in_result(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        result = mgr.read_events(r["run_id"])
        assert result["run_id"] == r["run_id"]


class TestStopRun:
    def test_transitions_queued_to_cancelled(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        stop = mgr.stop_run(r["run_id"])
        assert stop["status"] == "cancelled"
        assert stop["terminal"] is True

    def test_transitions_running_to_cancelled(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.transition_status(r["run_id"], "running")
        stop = mgr.stop_run(r["run_id"])
        assert stop["status"] == "cancelled"
        assert stop["terminal"] is True

    def test_unknown_run_returns_not_found(self):
        mgr = RunManager()
        result = mgr.stop_run("run_nonexistent")
        assert result.get("error") == "not_found"

    def test_terminal_run_returns_current_status(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.stop_run(r["run_id"])  # first stop -> cancelled
        result = mgr.stop_run(r["run_id"])  # second stop
        assert result["status"] == "cancelled"
        assert result.get("terminal") is True

    def test_stop_appends_done_event(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.stop_run(r["run_id"])
        events = mgr.read_events(r["run_id"])
        event_types = [e["type"] for e in events["events"]]
        assert EVENT_DONE in event_types

    def test_stop_removes_controls(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.stop_run(r["run_id"])
        result = mgr.stop_run(r["run_id"])
        assert result["controls"] == []


class TestAppendEvent:
    def test_appends_event_to_run(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        event = mgr.append_event(r["run_id"], "tool.started", payload={"tool": "test"})
        assert event is not None
        assert event.type == "tool.started"
        assert event.payload == {"tool": "test"}

    def test_unknown_run_returns_none(self):
        mgr = RunManager()
        assert mgr.append_event("run_nonexistent", "test") is None

    def test_event_updates_status(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.append_event(r["run_id"], "event.x")
        status = mgr.get_status(r["run_id"])
        assert status["last_seq"] == 2
        assert status["last_event_id"] == f"{r['run_id']}:2"

    def test_terminal_event_sets_status_terminal(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.append_event(r["run_id"], EVENT_DONE)
        status = mgr.get_status(r["run_id"])
        assert status["terminal"] is True


class TestStatusTransitions:
    def test_transition_status_changes_state(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        result = mgr.transition_status(r["run_id"], RUN_STATUS_RUNNING)
        assert result["status"] == "running"

    def test_transition_to_terminal(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        result = mgr.transition_status(r["run_id"], RUN_STATUS_CANCELLED)
        assert result["terminal"] is True

    def test_complete_run(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        result = mgr.complete_run(r["run_id"], result="done!")
        assert result["status"] == "completed"
        assert result["terminal"] is True
        assert result["result"] == "done!"

    def test_fail_run(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        result = mgr.fail_run(r["run_id"], error="something went wrong")
        assert result["status"] == "failed"
        assert result["terminal"] is True
        assert result["error"] == "something went wrong"

    def test_complete_after_stop_is_noop(self):
        mgr = RunManager()
        r = mgr.create_run("s")
        mgr.stop_run(r["run_id"])
        result = mgr.complete_run(r["run_id"])
        assert result["status"] == "cancelled"


class TestApprovalRequestAndResolve:
    def test_request_approval_adds_pending_id_and_events(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.request_approval(r["run_id"], "apr-1", payload={"command": "ls"})
        assert result is not None
        status = mgr.get_status(r["run_id"])
        assert "apr-1" in status["pending_approval_ids"]
        events = mgr.read_events(r["run_id"])
        types = [e["type"] for e in events["events"]]
        assert EVENT_APPROVAL_REQUESTED in types

    def test_resolve_approval_removes_pending_id_and_appends_event(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_approval(r["run_id"], "apr-1")
        result = mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        assert result.get("status") == "resolved"
        status = mgr.get_status(r["run_id"])
        assert "apr-1" not in status["pending_approval_ids"]
        events = mgr.read_events(r["run_id"])
        types = [e["type"] for e in events["events"]]
        assert EVENT_APPROVAL_RESOLVED in types

    def test_resolve_approval_unknown_run_returns_not_found(self):
        mgr = RunManager()
        result = mgr.resolve_approval("run_nonexistent", "apr-1", "approve")
        assert result["error"] == "not_found"

    def test_resolve_approval_unknown_id_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.resolve_approval(r["run_id"], "apr_unknown", "approve")
        assert result["error"] == "not_found"

    def test_resolve_approval_duplicate_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_approval(r["run_id"], "apr-1")
        mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        result = mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        assert result["error"] == "conflict"

    def test_resolve_approval_terminal_run_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_approval(r["run_id"], "apr-1")
        mgr.stop_run(r["run_id"])
        result = mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        assert result["error"] == "conflict"

    def test_request_clarify_adds_pending_id_and_events(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.request_clarify(r["run_id"], "clar-1", payload={"question": "OK?"})
        assert result is not None
        status = mgr.get_status(r["run_id"])
        assert "clar-1" in status["pending_clarify_ids"]

    def test_resolve_clarify_removes_pending_id_and_appends_event(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_clarify(r["run_id"], "clar-1")
        result = mgr.resolve_clarify(r["run_id"], "clar-1", "yes")
        assert result.get("status") == "resolved"
        status = mgr.get_status(r["run_id"])
        assert "clar-1" not in status["pending_clarify_ids"]

    def test_resolve_clarify_unknown_id_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.resolve_clarify(r["run_id"], "clar_unknown", "yes")
        assert result["error"] == "not_found"

    def test_clarify_duplicate_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_clarify(r["run_id"], "clar-1")
        mgr.resolve_clarify(r["run_id"], "clar-1", "yes")
        result = mgr.resolve_clarify(r["run_id"], "clar-1", "yes")
        assert result["error"] == "conflict"

    def test_clarify_terminal_run_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_clarify(r["run_id"], "clar-1")
        mgr.stop_run(r["run_id"])
        result = mgr.resolve_clarify(r["run_id"], "clar-1", "yes")
        assert result["error"] == "conflict"


class TestThreadSafety:
    def test_concurrent_creates_dont_corrupt(self):
        import threading

        mgr = RunManager()
        results = []
        errors = []

        def _create():
            try:
                r = mgr.create_run("sess_shared")
                results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        ids = {r["run_id"] for r in results}
        assert len(ids) == 10
