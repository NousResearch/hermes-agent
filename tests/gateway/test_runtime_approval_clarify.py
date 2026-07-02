"""Tests for approval/clarify lifecycle in RunManager and routes.

Covers:
- request_approval / request_clarify lifecycle
- resolve_approval / resolve_clarify full lifecycle
- Unknown run/action_id handling
- Duplicate resolution
- Terminal run conflict
- Secret redaction
- URL path run_id precedence
- Concurrent resolution
"""

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.models import (
    RUN_STATUS_QUEUED,
    RUN_STATUS_AWAITING_APPROVAL,
    RUN_STATUS_AWAITING_CLARIFY,
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
    EVENT_CLARIFY_REQUESTED,
    EVENT_CLARIFY_RESOLVED,
    redact_secrets,
)


class TestRequestApproval:
    def test_request_approval_adds_pending_approval_id(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.request_approval(r["run_id"], "apr-001", payload={"command": "echo ok"})
        assert result is not None
        assert result["approval_id"] == "apr-001"
        assert result["status"] == "requested"
        status = mgr.get_status(r["run_id"])
        assert "apr-001" in status["pending_approval_ids"]
        assert status["status"] == RUN_STATUS_AWAITING_APPROVAL

    def test_request_approval_appends_events(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001", payload={"command": "echo ok"})
        events = mgr.read_events(r["run_id"])
        types = [e["type"] for e in events["events"]]
        assert EVENT_APPROVAL_REQUESTED in types

    def test_request_approval_unknown_run_returns_none(self):
        mgr = RunManager()
        result = mgr.request_approval("run_nonexistent", "apr-001")
        assert result is None

    def test_request_approval_terminal_run_returns_none(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.stop_run(r["run_id"])
        result = mgr.request_approval(r["run_id"], "apr-001")
        assert result is None

    def test_request_approval_redacts_secrets(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001", payload={
            "command": "echo ok",
            "api_key": "secret-12345",
            "token": "tk-deadbeef",
        })
        events = mgr.read_events(r["run_id"])
        for ev in events["events"]:
            if ev["type"] == EVENT_APPROVAL_REQUESTED:
                p = ev["payload"].get("payload", {})
                assert p.get("api_key") == "<<redacted>>"
                assert p.get("token") == "<<redacted>>"
                assert p.get("command") == "echo ok"


class TestRequestClarify:
    def test_request_clarify_adds_pending_clarify_id(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.request_clarify(r["run_id"], "clar-001", payload={"question": "Proceed?"})
        assert result is not None
        assert result["clarify_id"] == "clar-001"
        assert result["status"] == "requested"
        status = mgr.get_status(r["run_id"])
        assert "clar-001" in status["pending_clarify_ids"]
        assert status["status"] == RUN_STATUS_AWAITING_CLARIFY

    def test_request_clarify_appends_events(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        events = mgr.read_events(r["run_id"])
        types = [e["type"] for e in events["events"]]
        assert EVENT_CLARIFY_REQUESTED in types

    def test_request_clarify_unknown_run_returns_none(self):
        mgr = RunManager()
        result = mgr.request_clarify("run_nonexistent", "clar-001")
        assert result is None

    def test_request_clarify_terminal_run_returns_none(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.stop_run(r["run_id"])
        result = mgr.request_clarify(r["run_id"], "clar-001")
        assert result is None

    def test_request_clarify_redacts_secrets(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001", payload={
            "question": "Proceed?",
            "password": "hunter2",
            "secret": "sssh",
        })
        events = mgr.read_events(r["run_id"])
        for ev in events["events"]:
            if ev["type"] == EVENT_CLARIFY_REQUESTED:
                p = ev["payload"].get("payload", {})
                assert p.get("password") == "<<redacted>>"
                assert p.get("secret") == "<<redacted>>"
                assert p.get("question") == "Proceed?"


class TestResolveApproval:
    def test_resolve_approval_removes_pending_approval_id(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        result = mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        assert result.get("status") == "resolved"
        status = mgr.get_status(r["run_id"])
        assert "apr-001" not in status["pending_approval_ids"]

    def test_resolve_approval_appends_resolved_event(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        result = mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        assert result["type"] == "approval"
        assert result["status"] == "resolved"
        assert "event" in result
        event = result["event"]
        assert event["type"] == EVENT_APPROVAL_RESOLVED
        assert event["payload"]["approval_id"] == "apr-001"
        assert event["payload"]["choice"] == "approve"

    def test_resolve_approval_returns_resolved_response(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        result = mgr.resolve_approval(r["run_id"], "apr-001", "deny")
        assert result["run_id"] == r["run_id"]
        assert result["action_id"] == "apr-001"
        assert result["type"] == "approval"
        assert result["status"] == "resolved"

    def test_resolve_approval_unknown_run_returns_not_found(self):
        mgr = RunManager()
        result = mgr.resolve_approval("run_nonexistent", "apr-001", "approve")
        assert result["error"] == "not_found"

    def test_resolve_approval_unknown_approval_id_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.resolve_approval(r["run_id"], "apr_nonexistent", "approve")
        assert result["error"] == "not_found"

    def test_resolve_approval_duplicate_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        result = mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        assert result["error"] == "conflict"

    def test_resolve_approval_terminal_run_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        mgr.stop_run(r["run_id"])
        status = mgr.get_status(r["run_id"])
        assert status["terminal"] is True
        result = mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        assert result["error"] == "conflict"

    def test_resolve_approval_redacts_secrets_in_event_payload(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        result = mgr.resolve_approval(r["run_id"], "apr-001", "approve", payload={
            "api_key": "should-be-redacted",
            "note": "safe",
        })
        event = result["event"]
        p = event["payload"].get("payload", {})
        assert p.get("api_key") == "<<redacted>>"
        assert p.get("note") == "safe"

    def test_resolve_approval_transitions_back_to_queued_when_no_pending_left(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        status = mgr.get_status(r["run_id"])
        assert status["status"] == RUN_STATUS_QUEUED

    def test_resolve_approval_stays_awaiting_with_multiple_pending(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        mgr.request_approval(r["run_id"], "apr-002")
        mgr.resolve_approval(r["run_id"], "apr-001", "approve")
        status = mgr.get_status(r["run_id"])
        assert status["status"] == RUN_STATUS_AWAITING_APPROVAL
        assert "apr-002" in status["pending_approval_ids"]

    def test_approval_without_request_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.resolve_approval(r["run_id"], "apr_nonexistent", "approve")
        assert result["error"] == "not_found"


class TestResolveClarify:
    def test_resolve_clarify_removes_pending_clarify_id(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        result = mgr.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result.get("status") == "resolved"
        status = mgr.get_status(r["run_id"])
        assert "clar-001" not in status["pending_clarify_ids"]

    def test_resolve_clarify_appends_resolved_event(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        result = mgr.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result["type"] == "clarify"
        assert result["status"] == "resolved"
        event = result["event"]
        assert event["type"] == EVENT_CLARIFY_RESOLVED
        assert event["payload"]["clarify_id"] == "clar-001"
        assert event["payload"]["answer"] == "yes"

    def test_resolve_clarify_unknown_clarify_id_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.resolve_clarify(r["run_id"], "clar_nonexistent", "yes")
        assert result["error"] == "not_found"

    def test_clarify_duplicate_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        mgr.resolve_clarify(r["run_id"], "clar-001", "yes")
        result = mgr.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result["error"] == "conflict"

    def test_clarify_terminal_run_returns_conflict(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        mgr.stop_run(r["run_id"])
        result = mgr.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result["error"] == "conflict"

    def test_resolve_clarify_redacts_secrets_in_event_payload(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        result = mgr.resolve_clarify(r["run_id"], "clar-001", "yes", payload={
            "access_token": "should-be-redacted",
            "note": "public",
        })
        event = result["event"]
        p = event["payload"].get("payload", {})
        assert p.get("access_token") == "<<redacted>>"
        assert p.get("note") == "public"

    def test_resolve_clarify_transitions_back_when_no_pending_left(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        mgr.resolve_clarify(r["run_id"], "clar-001", "yes")
        status = mgr.get_status(r["run_id"])
        assert status["status"] == RUN_STATUS_QUEUED


class TestSecretRedaction:
    def test_payload_secrets_are_redacted_on_request_approval(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-sec", payload={
            "api_key": "sk-1234",
            "apikey": "key-5678",
            "token": "tk-9012",
            "access_token": "at-3456",
            "refresh_token": "rt-7890",
            "password": "p4ssw0rd",
            "secret": "s3cret",
            "authorization": "bearer xyz",
            "bearer": "b-123",
            "normal_field": "keep me",
        })
        events = mgr.read_events(r["run_id"])
        for ev in events["events"]:
            if ev["type"] == EVENT_APPROVAL_REQUESTED:
                p = ev["payload"].get("payload", {})
                assert p.get("api_key") == "<<redacted>>"
                assert p.get("apikey") == "<<redacted>>"
                assert p.get("token") == "<<redacted>>"
                assert p.get("access_token") == "<<redacted>>"
                assert p.get("refresh_token") == "<<redacted>>"
                assert p.get("password") == "<<redacted>>"
                assert p.get("secret") == "<<redacted>>"
                assert p.get("authorization") == "<<redacted>>"
                assert p.get("bearer") == "<<redacted>>"
                assert p.get("normal_field") == "keep me"

    def test_payload_secrets_are_redacted_on_request_clarify(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-sec", payload={
            "api_key": "sk-1234",
            "token": "tk-9012",
            "password": "p4ssw0rd",
            "normal": "visible",
        })
        events = mgr.read_events(r["run_id"])
        for ev in events["events"]:
            if ev["type"] == EVENT_CLARIFY_REQUESTED:
                p = ev["payload"].get("payload", {})
                assert p.get("api_key") == "<<redacted>>"
                assert p.get("token") == "<<redacted>>"
                assert p.get("password") == "<<redacted>>"
                assert p.get("normal") == "visible"

    def test_resolution_event_payload_is_redacted(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-redact")
        result = mgr.resolve_approval(r["run_id"], "apr-redact", "approve", payload={
            "api_key": "sk-secret",
        })
        event = result["event"]
        assert "sk-secret" not in str(event)


class TestURLPathRunIdWins:
    def test_resolve_approval_uses_callers_run_id_not_body(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-url")
        result = mgr.resolve_approval(r["run_id"], "apr-url", "approve")
        assert result["status"] == "resolved"

    def test_resolve_clarify_uses_callers_run_id_not_body(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-url")
        result = mgr.resolve_clarify(r["run_id"], "clar-url", "yes")
        assert result["status"] == "resolved"


class TestConcurrentResolution:
    def test_concurrent_approval_first_wins_second_gets_conflict(self):
        import threading

        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-conc")
        results = []

        def _resolve():
            res = mgr.resolve_approval(r["run_id"], "apr-conc", "approve")
            results.append(res)

        threads = [threading.Thread(target=_resolve) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        resolved = [res for res in results if res.get("status") == "resolved"]
        conflicts = [res for res in results if res.get("error") == "conflict"]
        assert len(resolved) == 1
        assert len(conflicts) == 3

    def test_concurrent_clarify_first_wins_second_gets_conflict(self):
        import threading

        mgr = RunManager()
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-conc")
        results = []

        def _resolve():
            res = mgr.resolve_clarify(r["run_id"], "clar-conc", "yes")
            results.append(res)

        threads = [threading.Thread(target=_resolve) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        resolved = [res for res in results if res.get("status") == "resolved"]
        conflicts = [res for res in results if res.get("error") == "conflict"]
        assert len(resolved) == 1
        assert len(conflicts) == 3


class TestBackwardCompatibility:
    def test_existing_approval_unsupported_is_removed(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.resolve_approval(r["run_id"], "apr-nonexistent", "approve")
        assert result["error"] == "not_found"

    def test_existing_clarify_unsupported_is_removed(self):
        mgr = RunManager()
        r = mgr.create_run("sess_1")
        result = mgr.resolve_clarify(r["run_id"], "clar-nonexistent", "yes")
        assert result["error"] == "not_found"

    def test_unknown_run_consistently_returns_not_found(self):
        mgr = RunManager()
        assert mgr.resolve_approval("run_ghost", "apr-g", "approve")["error"] == "not_found"
        assert mgr.resolve_clarify("run_ghost", "clar-g", "yes")["error"] == "not_found"
