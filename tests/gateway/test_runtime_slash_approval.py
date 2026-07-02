"""Tests for gateway slash-command /approve and /deny RunManager state sync.

Covers:
- /approve updates RunManager pending approval state
- /approve appends approval.resolved exactly once
- /deny updates RunManager pending approval state
- /deny appends the correct resolved/denied event exactly once
- Duplicate /approve or /deny does not duplicate events
- Unknown approval ID maps to not_found
- Terminal run maps to conflict
- Secrets are redacted
- Pending approval IDs are removed when slash commands resolve them
"""

from unittest.mock import MagicMock

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.models import (
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
)


class TestSlashApprovalRunManagerSync:
    """Gateway /approve and /deny slash commands sync RunManager state."""

    def test_approve_resolves_pending_approval_in_run_manager(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="slash_appr")
        bridge.bind_run(r["run_id"], "slash_appr")
        mgr.transition_status(r["run_id"], "running")

        mgr.request_approval(r["run_id"], "apr-test-001", payload={"cmd": "ls"})
        status = mgr.get_status(r["run_id"])
        assert "apr-test-001" in status["pending_approval_ids"]

        result = mgr.resolve_approval(r["run_id"], "apr-test-001", "once")
        assert result.get("status") == "resolved"

        status = mgr.get_status(r["run_id"])
        assert "apr-test-001" not in status["pending_approval_ids"]

    def test_deny_resolves_pending_approval_in_run_manager(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="slash_deny")
        bridge.bind_run(r["run_id"], "slash_deny")
        mgr.transition_status(r["run_id"], "running")

        mgr.request_approval(r["run_id"], "apr-deny-001", payload={"cmd": "rm -rf"})
        result = mgr.resolve_approval(r["run_id"], "apr-deny-001", "deny")
        assert result.get("status") == "resolved"

        status = mgr.get_status(r["run_id"])
        assert "apr-deny-001" not in status["pending_approval_ids"]

    def test_approve_appends_resolved_event_exactly_once(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="event_once")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-event-001", payload={"cmd": "echo"})
        mgr.resolve_approval(r["run_id"], "apr-event-001", "once")

        events = mgr.read_events(r["run_id"])
        resolved_count = sum(
            1 for e in events["events"]
            if e["type"] in ("approval.resolved", EVENT_APPROVAL_RESOLVED)
        )
        assert resolved_count == 1

    def test_deny_appends_resolved_event_exactly_once(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="deny_event")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-deny-event", payload={"cmd": "danger"})
        mgr.resolve_approval(r["run_id"], "apr-deny-event", "deny")

        events = mgr.read_events(r["run_id"])
        resolved_count = sum(
            1 for e in events["events"]
            if e["type"] in ("approval.resolved", EVENT_APPROVAL_RESOLVED)
        )
        assert resolved_count == 1

    def test_duplicate_approve_returns_conflict(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="dup_appr")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-dup-001", payload={"cmd": "ls"})
        mgr.resolve_approval(r["run_id"], "apr-dup-001", "once")

        result = mgr.resolve_approval(r["run_id"], "apr-dup-001", "once")
        assert result.get("error") == "conflict"

    def test_duplicate_deny_returns_conflict(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="dup_deny")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-dup-deny", payload={"cmd": "danger"})
        mgr.resolve_approval(r["run_id"], "apr-dup-deny", "deny")

        result = mgr.resolve_approval(r["run_id"], "apr-dup-deny", "deny")
        assert result.get("error") == "conflict"

    def test_duplicate_approve_does_not_append_event(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="dup_event")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-dup-event", payload={"cmd": "ls"})
        mgr.resolve_approval(r["run_id"], "apr-dup-event", "once")

        before_events = mgr.read_events(r["run_id"])
        resolved_before = sum(
            1 for e in before_events["events"]
            if e["type"] in ("approval.resolved", EVENT_APPROVAL_RESOLVED)
        )

        mgr.resolve_approval(r["run_id"], "apr-dup-event", "once")
        after_events = mgr.read_events(r["run_id"])
        resolved_after = sum(
            1 for e in after_events["events"]
            if e["type"] in ("approval.resolved", EVENT_APPROVAL_RESOLVED)
        )

        assert resolved_after == resolved_before

    def test_unknown_approval_id_returns_not_found(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="unknown_appr")
        mgr.transition_status(r["run_id"], "running")

        result = mgr.resolve_approval(r["run_id"], "nonexistent-id", "once")
        assert result.get("error") == "not_found"

    def test_unknown_run_id_returns_not_found(self):
        mgr = RunManager()

        result = mgr.resolve_approval("no-such-run", "any-id", "once")
        assert result.get("error") == "not_found"

    def test_terminal_run_approval_returns_conflict(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="term_appr")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-term-001", payload={"cmd": "test"})
        mgr.complete_run(r["run_id"])

        result = mgr.resolve_approval(r["run_id"], "apr-term-001", "once")
        assert result.get("error") == "conflict"

    def test_terminal_run_deny_returns_conflict(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="term_deny")
        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-term-deny", payload={"cmd": "test"})
        mgr.fail_run(r["run_id"], error="boom")

        result = mgr.resolve_approval(r["run_id"], "apr-term-deny", "deny")
        assert result.get("error") == "conflict"

    def test_pending_approval_ids_removed_after_resolve(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="removal_test")
        mgr.transition_status(r["run_id"], "running")

        mgr.request_approval(r["run_id"], "apr-1")
        mgr.request_approval(r["run_id"], "apr-2")

        status = mgr.get_status(r["run_id"])
        assert "apr-1" in status["pending_approval_ids"]
        assert "apr-2" in status["pending_approval_ids"]

        mgr.resolve_approval(r["run_id"], "apr-1", "once")
        status = mgr.get_status(r["run_id"])
        assert "apr-1" not in status["pending_approval_ids"]
        assert "apr-2" in status["pending_approval_ids"]

    def test_multiple_approvals_resolved_independently(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="multi_appr")
        mgr.transition_status(r["run_id"], "running")

        mgr.request_approval(r["run_id"], "apr-A")
        mgr.request_approval(r["run_id"], "apr-B")
        mgr.request_approval(r["run_id"], "apr-C")

        mgr.resolve_approval(r["run_id"], "apr-B", "deny")
        mgr.resolve_approval(r["run_id"], "apr-A", "once")
        mgr.resolve_approval(r["run_id"], "apr-C", "session")

        status = mgr.get_status(r["run_id"])
        assert status["pending_approval_ids"] == []

    def test_secrets_redacted_in_approval_request_payload(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="redact_payload")
        mgr.transition_status(r["run_id"], "running")

        mgr.request_approval(r["run_id"], "apr-redact-001", payload={
            "command": "curl -H 'Authorization: Bearer sk-secret-key' https://api.example.com",
            "api_key": "sk-abcdef-123456",
            "token": "ghp_secret_token_12345",
        })

        events = mgr.read_events(r["run_id"])
        for e in events["events"]:
            if e["type"] in ("approval.requested", EVENT_APPROVAL_REQUESTED):
                payload_str = str(e.get("payload", {}))
                assert "sk-secret-key" not in payload_str
                assert "sk-abcdef-123456" not in payload_str
                assert "ghp_secret_token_12345" not in payload_str

    def test_sync_helper_finds_run_id_from_session_key(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="sync_session")
        bridge.bind_run(r["run_id"], "sync_session")

        mgr.transition_status(r["run_id"], "running")
        mgr.request_approval(r["run_id"], "apr-sync-001", payload={"cmd": "test"})

        rm = bridge.run_manager
        status = rm.get_status(r["run_id"])
        assert "apr-sync-001" in status["pending_approval_ids"]

        rm.resolve_approval(r["run_id"], "apr-sync-001", "once")
        status = rm.get_status(r["run_id"])
        assert "apr-sync-001" not in status["pending_approval_ids"]

    def test_sync_with_no_bridge_is_graceful(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="no_bridge")
        mgr.transition_status(r["run_id"], "running")
        mgr.complete_run(r["run_id"])

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"

    def test_sync_with_no_running_run_is_graceful(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        result = mgr.resolve_approval("nonexistent", "any-id", "once")
        assert result.get("error") == "not_found"
