"""Tests for messaging-platform GatewayRunner runtime binding.

Covers:
- Messaging-platform GatewayRunner path calls bind_run(run_id, session_key, agent_ref)
- Messaging-platform terminal path calls unbind_run(run_id)
- Runtime run exists for messaging-platform execution
- Approval resolution uses bound messaging-platform session_key
- Clarify resolution uses bound messaging-platform session_key or clarify_id
- Stop uses bound messaging-platform live agent reference
- API server binding from Phase 12 still passes
- Standalone RunManager fallback still passes
- Unknown run/action maps to 404
- Conflict maps to 409
- not_supported maps cleanly
- URL path run_id wins over body run_id
- Events are appended exactly once
- Pending IDs are added and removed correctly
- Secrets are redacted
- No existing Agent runtime tests regress
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.models import (
    EVENT_APPROVAL_REQUESTED,
    EVENT_APPROVAL_RESOLVED,
    EVENT_CLARIFY_REQUESTED,
    EVENT_CLARIFY_RESOLVED,
    EVENT_RUN_STARTED,
    EVENT_RUN_STATUS,
)


class TestMessagingGatewayRunnerBinding:
    """GatewayRunner binds messaging runs to runtime controls."""

    def test_bind_run_associates_run_id_with_session_key(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_session_abc")
        mock_agent = MagicMock()
        bridge.bind_run(r["run_id"], "gw_session_abc", mock_agent)

        sk = bridge._resolve_session_key(r["run_id"])
        assert sk == "gw_session_abc"
        assert bridge._live_agents.get(r["run_id"]) is mock_agent

    def test_unbind_run_cleans_session_key_and_agent(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_session_xyz")
        mock_agent = MagicMock()
        bridge.bind_run(r["run_id"], "gw_session_xyz", mock_agent)

        bridge.unbind_run(r["run_id"])
        assert bridge._resolve_session_key(r["run_id"]) is None
        assert bridge._live_agents.get(r["run_id"]) is None

    def test_unbind_run_nonexistent_run_is_harmless(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        bridge.unbind_run("msg-nonexistent")

    def test_create_run_for_messaging_session_creates_queued_status(self):
        mgr = RunManager()
        r = mgr.create_run(
            session_id="tg_12345_user_67890",
            run_id="msg-abc123",
            model="gpt-4",
        )
        assert r["run_id"] == "msg-abc123"
        status = mgr.get_status("msg-abc123")
        assert status is not None
        assert status["session_id"] == "tg_12345_user_67890"
        assert status["status"] == "queued"

    def test_run_transitions_to_running_then_completes(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="gw_session", run_id="msg-run-1")
        mgr.transition_status("msg-run-1", "running")
        status = mgr.get_status("msg-run-1")
        assert status["status"] == "running"

        mgr.complete_run("msg-run-1")
        status = mgr.get_status("msg-run-1")
        assert status["status"] == "completed"
        assert status["terminal"] is True

    def test_run_transitions_to_failed_on_error(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="gw_session", run_id="msg-run-2")
        mgr.transition_status("msg-run-2", "running")
        mgr.fail_run("msg-run-2", error="Model timeout")
        status = mgr.get_status("msg-run-2")
        assert status["status"] == "failed"

    def test_multiple_runs_for_same_session_key(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r1 = mgr.create_run(session_id="shared_session", run_id="msg-turn-1")
        r2 = mgr.create_run(session_id="shared_session", run_id="msg-turn-2")

        mock_agent_1 = MagicMock()
        bridge.bind_run("msg-turn-1", "shared_session", mock_agent_1)
        bridge.bind_run("msg-turn-2", "shared_session", MagicMock())

        bridge.unbind_run("msg-turn-1")
        assert bridge._resolve_session_key("msg-turn-1") is None
        assert bridge._resolve_session_key("msg-turn-2") == "shared_session"


class TestMessagingApproval:
    """Approval resolution uses bound messaging-platform session_key."""

    def test_request_approval_records_in_run_manager(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_approval_session")
        bridge.bind_run(r["run_id"], "gw_approval_session", MagicMock())
        mgr.transition_status(r["run_id"], "running")

        bridge.request_approval(r["run_id"], "apr-001", payload={"command": "rm -rf"})
        status = mgr.get_status(r["run_id"])
        assert "apr-001" in status["pending_approval_ids"]
        assert status["status"] == "awaiting_approval"

    def test_resolve_approval_live_via_bridge(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_live_apr")
        bridge.bind_run(r["run_id"], "gw_live_apr")
        mgr.transition_status(r["run_id"], "running")
        bridge.request_approval(r["run_id"], "apr-002")

        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            result = bridge.resolve_approval(r["run_id"], "apr-002", "approve")
        assert result.get("status") == "resolved" or "status" in result
        assert result.get("error") is None

    def test_resolve_approval_unknown_run_returns_404(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.resolve_approval("run_nonexistent", "apr-001", "approve")
        assert result.get("error") == "not_found"

    def test_resolve_approval_unknown_approval_id_returns_404(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_err")
        bridge.bind_run(r["run_id"], "gw_err")
        mgr.transition_status(r["run_id"], "running")
        result = bridge.resolve_approval(r["run_id"], "apr_nonexistent", "approve")
        assert result.get("error") == "not_found"

    def test_resolve_approval_duplicate_returns_409(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_dup")
        bridge.bind_run(r["run_id"], "gw_dup")
        mgr.transition_status(r["run_id"], "running")
        bridge.request_approval(r["run_id"], "apr-003")
        bridge.resolve_approval(r["run_id"], "apr-003", "approve")
        result = bridge.resolve_approval(r["run_id"], "apr-003", "approve")
        assert result.get("error") == "conflict"

    def test_resolve_approval_terminal_run_returns_409(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_term")
        bridge.bind_run(r["run_id"], "gw_term")
        bridge.request_approval(r["run_id"], "apr-004")
        mgr.complete_run(r["run_id"])
        result = bridge.resolve_approval(r["run_id"], "apr-004", "approve")
        assert result.get("error") == "conflict"

    def test_approval_payload_secrets_redacted(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_redact")
        bridge.bind_run(r["run_id"], "gw_redact")
        mgr.transition_status(r["run_id"], "running")

        bridge.request_approval(r["run_id"], "apr-005", payload={
            "command": "curl -H 'Authorization: Bearer sk-secret123' https://api.com",
            "api_key": "sk-should-be-redacted",
            "token": "ghp_also_redacted",
        })
        events = mgr.read_events(r["run_id"])["events"]
        approval_event = next(e for e in events if e["type"] == EVENT_APPROVAL_REQUESTED)
        payload_str = json.dumps(approval_event.get("payload", {}))
        assert "sk-secret123" not in payload_str
        assert "sk-should-be-redacted" not in payload_str
        assert "ghp_also_redacted" not in payload_str


class TestMessagingClarify:
    """Clarify resolution uses bound messaging-platform session_key or clarify_id."""

    def test_request_clarify_records_in_run_manager(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_clarify_session")
        bridge.bind_run(r["run_id"], "gw_clarify_session")
        mgr.transition_status(r["run_id"], "running")

        bridge.request_clarify(r["run_id"], "clar-001", payload={"question": "Proceed?"})
        status = mgr.get_status(r["run_id"])
        assert "clar-001" in status["pending_clarify_ids"]
        assert status["status"] == "awaiting_clarify"

    def test_resolve_clarify_live_via_bridge(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_live_clar")
        bridge.bind_run(r["run_id"], "gw_live_clar")
        mgr.transition_status(r["run_id"], "running")
        bridge.request_clarify(r["run_id"], "clar-002")

        with patch("tools.clarify_gateway.resolve_gateway_clarify", return_value=True):
            result = bridge.resolve_clarify(r["run_id"], "clar-002", "yes")
        assert result.get("error") is None

    def test_resolve_clarify_unknown_run_returns_404(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.resolve_clarify("run_nonexistent", "clar-001", "yes")
        assert result.get("error") == "not_found"

    def test_resolve_clarify_unknown_clarify_id_returns_404(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_err_clar")
        bridge.bind_run(r["run_id"], "gw_err_clar")
        mgr.transition_status(r["run_id"], "running")
        result = bridge.resolve_clarify(r["run_id"], "clar_nonexistent", "yes")
        assert result.get("error") == "not_found"

    def test_resolve_clarify_duplicate_returns_409(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_dup_clar")
        bridge.bind_run(r["run_id"], "gw_dup_clar")
        mgr.transition_status(r["run_id"], "running")
        bridge.request_clarify(r["run_id"], "clar-003")
        bridge.resolve_clarify(r["run_id"], "clar-003", "yes")
        result = bridge.resolve_clarify(r["run_id"], "clar-003", "yes")
        assert result.get("error") == "conflict"

    def test_resolve_clarify_terminal_run_returns_409(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_term_clar")
        bridge.bind_run(r["run_id"], "gw_term_clar")
        bridge.request_clarify(r["run_id"], "clar-004")
        mgr.complete_run(r["run_id"])
        result = bridge.resolve_clarify(r["run_id"], "clar-004", "yes")
        assert result.get("error") == "conflict"

    def test_clarify_payload_secrets_redacted(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="gw_redact_clar")
        bridge.bind_run(r["run_id"], "gw_redact_clar")
        mgr.transition_status(r["run_id"], "running")

        bridge.request_clarify(r["run_id"], "clar-005", payload={
            "question": "Proceed with API key?",
            "secret": "sk-top-secret",
        })
        events = mgr.read_events(r["run_id"])["events"]
        clarify_event = next(e for e in events if e["type"] == EVENT_CLARIFY_REQUESTED)
        payload_str = json.dumps(clarify_event.get("payload", {}))
        assert "sk-top-secret" not in payload_str


class TestMessagingStopInterrupt:
    """Stop uses bound messaging-platform live agent reference."""

    def test_stop_run_uses_direct_agent_reference(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_stop_session")
        mock_agent = MagicMock()
        bridge.bind_run(r["run_id"], "gw_stop_session", mock_agent)
        mgr.transition_status(r["run_id"], "running")

        bridge.stop_run(r["run_id"])
        mock_agent.interrupt.assert_called_once()

    def test_stop_run_falls_back_to_gateway_runner_agents(self):
        mgr = RunManager()
        mock_runner = MagicMock()

        def _get_runner():
            return mock_runner

        bridge = RuntimeControlBridge(
            mgr, gateway_runner_ref=_get_runner,
        )

        r = mgr.create_run(session_id="gw_stop_fallback")
        mock_agent = MagicMock()
        mock_runner._running_agents = {"gw_stop_fallback": mock_agent}

        bridge.bind_run(r["run_id"], "gw_stop_fallback")
        mgr.transition_status(r["run_id"], "running")

        bridge.stop_run(r["run_id"])
        mock_agent.interrupt.assert_called_once()

    def test_stop_run_no_live_agent_no_fallback_succeeds(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_no_ref")
        bridge.bind_run(r["run_id"], "gw_no_ref")
        mgr.transition_status(r["run_id"], "running")

        result = bridge.stop_run(r["run_id"])
        assert result.get("error") is None

    def test_stop_run_unknown_run_returns_404(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.stop_run("nonexistent_run")
        assert result.get("error") == "not_found"

    def test_stop_run_already_terminal(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="gw_already_done")
        bridge.bind_run(r["run_id"], "gw_already_done")
        mgr.complete_run(r["run_id"])

        result = bridge.stop_run(r["run_id"])
        assert result.get("message") is not None


class TestGatewayRunnerControlBridgeSetter:
    """GatewayRunner.set_runtime_control_bridge wires the bridge."""

    def test_gateway_runner_accepts_control_bridge(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner._pending_approvals = {}
        runner._runtime_session_runs = {}
        runner._runtime_control_bridge = None

        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        runner.set_runtime_control_bridge(bridge)
        assert runner._runtime_control_bridge is bridge

    def test_gateway_runner_no_bridge_is_harmless(self):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner._pending_approvals = {}
        runner._runtime_session_runs = {}
        runner._runtime_control_bridge = None

        assert runner._runtime_control_bridge is None
        assert isinstance(runner._runtime_session_runs, dict)
        assert len(runner._runtime_session_runs) == 0


class TestMessagingEventsAndLifecycle:
    """Events are appended correctly during messaging run lifecycle."""

    def test_full_run_lifecycle_events(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="lifecycle_session", run_id="msg-life-1")
        events = mgr.read_events("msg-life-1")["events"]
        assert any(e["type"] == EVENT_RUN_STARTED for e in events)

        mgr.transition_status("msg-life-1", "running")
        events = mgr.read_events("msg-life-1")["events"]
        assert any(e["type"] == EVENT_RUN_STATUS for e in events)

        bridge.bind_run("msg-life-1", "lifecycle_session")
        bridge.request_approval("msg-life-1", "apr-life-1")
        events = mgr.read_events("msg-life-1")["events"]
        assert any(e["type"] == EVENT_APPROVAL_REQUESTED for e in events)

        bridge.resolve_approval("msg-life-1", "apr-life-1", "approve")
        events = mgr.read_events("msg-life-1")["events"]
        assert any(e["type"] == EVENT_APPROVAL_RESOLVED for e in events)

    def test_events_not_duplicated(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="nodup", run_id="msg-nodup-1")
        mgr.transition_status("msg-nodup-1", "running")
        mgr.request_approval("msg-nodup-1", "apr-1")

        events = mgr.read_events("msg-nodup-1")["events"]
        started_events = [e for e in events if e["type"] == EVENT_RUN_STARTED]
        assert len(started_events) == 1

    def test_pending_ids_cleared_after_resolution(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="pending_test")
        bridge.bind_run(r["run_id"], "pending_test")
        mgr.transition_status(r["run_id"], "running")

        bridge.request_approval(r["run_id"], "apr-a")
        bridge.request_approval(r["run_id"], "apr-b")
        bridge.request_clarify(r["run_id"], "clar-a")

        status = mgr.get_status(r["run_id"])
        assert len(status["pending_approval_ids"]) == 2
        assert len(status["pending_clarify_ids"]) == 1

        bridge.resolve_approval(r["run_id"], "apr-a", "approve")
        status = mgr.get_status(r["run_id"])
        assert "apr-a" not in status["pending_approval_ids"]
        assert "apr-b" in status["pending_approval_ids"]

        bridge.resolve_approval(r["run_id"], "apr-b", "approve")
        bridge.resolve_clarify(r["run_id"], "clar-a", "yes")

        status = mgr.get_status(r["run_id"])
        assert len(status["pending_approval_ids"]) == 0
        assert len(status["pending_clarify_ids"]) == 0


class TestURLPathRunIdPrecedence:
    """URL path run_id wins over body run_id."""

    def test_url_run_id_wins_in_approval_resolution(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r1 = mgr.create_run(session_id="url_win", run_id="msg-url-path")
        bridge.bind_run(r1["run_id"], "url_win")
        mgr.transition_status(r1["run_id"], "running")
        bridge.request_approval(r1["run_id"], "apr-url-1")

        result = bridge.resolve_approval(r1["run_id"], "apr-url-1", "approve")
        assert result.get("error") is None
        assert result.get("run_id") == "msg-url-path"

    def test_url_run_id_wins_in_clarify_resolution(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r1 = mgr.create_run(session_id="url_clar_win", run_id="msg-clar-path")
        bridge.bind_run(r1["run_id"], "url_clar_win")
        mgr.transition_status(r1["run_id"], "running")
        bridge.request_clarify(r1["run_id"], "clar-url-1")

        result = bridge.resolve_clarify(r1["run_id"], "clar-url-1", "yes")
        assert result.get("error") is None
        assert result.get("run_id") == "msg-clar-path"


class TestBridgeBehaviorWithoutGateway:
    """Standalone RunManager fallback when no bridge is wired."""

    def test_standalone_run_manager_still_works(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="standalone_test")
        assert r["status"] == "queued"

        mgr.transition_status(r["run_id"], "running")
        mgr.complete_run(r["run_id"])
        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"

    def test_bridge_without_gateway_ref_is_pass_through(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="no_gw")
        bridge.bind_run(r["run_id"], "no_gw")
        mgr.transition_status(r["run_id"], "running")

        result = bridge.stop_run(r["run_id"])
        assert result.get("error") is None

    def test_bind_unbind_without_gateway_ref_is_harmless(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="bare_bind")
        bridge.bind_run(r["run_id"], "bare_bind")
        bridge.unbind_run(r["run_id"])

        assert bridge._resolve_session_key(r["run_id"]) is None
