"""Tests for gateway.runtime.control_bridge.RuntimeControlBridge.

Covers:
- Pure RunManager passthrough (no bridge)
- Bridge with explicit run_id -> session_key bindings
- Bridge with callable session-key resolver
- resolve_approval delegates to live gateway approval
- resolve_clarify delegates to live gateway clarify
- stop_run delegates to live agent interrupt
- Error propagation (not_found, conflict)
- Secret redaction
- Bridge falls back gracefully when no live primitives exist
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge


class TestBridgePassthrough:
    """Bridge without live bindings behaves identically to RunManager."""

    def test_resolve_approval_passthrough_success(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
        assert result["status"] == "resolved"
        assert result["type"] == "approval"

    def test_resolve_approval_passthrough_not_found(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.resolve_approval("nonexistent", "apr-001", "once")
        assert result["error"] == "not_found"

    def test_resolve_approval_passthrough_conflict(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        bridge.resolve_approval(r["run_id"], "apr-001", "once")
        result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
        assert result["error"] == "conflict"

    def test_resolve_clarify_passthrough_success(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        result = bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result["status"] == "resolved"
        assert result["type"] == "clarify"

    def test_resolve_clarify_passthrough_not_found(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.resolve_clarify("nonexistent", "clar-001", "yes")
        assert result["error"] == "not_found"

    def test_stop_run_passthrough_success(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        assert result["terminal"] is True

    def test_stop_run_passthrough_not_found(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        result = bridge.stop_run("nonexistent")
        assert result["error"] == "not_found"

    def test_request_approval_passthrough(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        result = bridge.request_approval(r["run_id"], "apr-001")
        assert result is not None
        assert result["approval_id"] == "apr-001"

    def test_request_clarify_passthrough(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        result = bridge.request_clarify(r["run_id"], "clar-001")
        assert result is not None
        assert result["clarify_id"] == "clar-001"


class TestBridgeWithBindings:
    """Bridge with explicit run_id -> session_key bindings."""

    def test_bind_resolves_session_key(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_sk_abc123")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            mock_resolve.return_value = 1
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("gw_sk_abc123", "once")

    def test_stop_run_with_binding_interrupts_agent(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_runner = MagicMock()
        mock_runner._running_agents = {"gw_sk_stop": mock_agent}

        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "gw_sk_stop",
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_stop")
        bridge.bind_run(r["run_id"], "gw_sk_stop")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_called_once_with("run_stop")

    def test_stop_run_missing_agent_does_not_raise(self):
        mgr = RunManager()
        mock_runner = MagicMock()
        mock_runner._running_agents = {}

        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "unknown_session",
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_stop")
        bridge.bind_run(r["run_id"], "unknown_session")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"


class TestBridgeClarifyLive:
    """Bridge clarify resolution also delegates to live gateway."""

    def test_resolve_clarify_calls_live_primitive(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            mock_resolve.return_value = True
            result = bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("clar-001", "yes")

    def test_resolve_clarify_live_failure_still_succeeds(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001")
        with patch("tools.clarify_gateway.resolve_gateway_clarify") as mock_resolve:
            mock_resolve.side_effect = RuntimeError("not found")
            result = bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
            assert result["status"] == "resolved"


class TestBridgeApprovalLive:
    """Bridge approval resolution delegates to live gateway when session_key is known."""

    def test_resolve_approval_live_when_session_bound(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "sk_live_1",
        )
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            mock_resolve.return_value = 1
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock_resolve.assert_called_once_with("sk_live_1", "once")

    def test_resolve_approval_missing_session_key_still_succeeds(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
        assert result["status"] == "resolved"

    def test_resolve_approval_live_failure_still_succeeds(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "sk_live_1",
        )
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            mock_resolve.side_effect = RuntimeError("not found")
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"


class TestBridgeSecretRedaction:
    """Bridge does not expose secrets in results."""

    def test_approval_payload_redacted(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001", payload={
            "command": "echo ok",
            "api_key": "sk-secret-12345",
            "token": "tk-deadbeef",
        })
        result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
        assert result["status"] == "resolved"
        event = result.get("event", {})
        raw_payload = event.get("payload", {}).get("payload", {})
        assert raw_payload.get("api_key") != "sk-secret-12345"

    def test_clarify_payload_redacted(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_clarify(r["run_id"], "clar-001", payload={
            "question": "Proceed?",
            "password": "super-secret",
        })
        result = bridge.resolve_clarify(r["run_id"], "clar-001", "yes")
        assert result["status"] == "resolved"
        event = result.get("event", {})
        raw_payload = event.get("payload", {}).get("payload", {})
        assert raw_payload.get("password") != "super-secret"


class TestBridgeStopLive:
    """Bridge stop delegates to live agent interrupt when available."""

    def test_stop_run_with_live_agent_calls_interrupt(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_runner = MagicMock()
        mock_runner._running_agents = {"test_session": mock_agent}

        bridge = RuntimeControlBridge(
            mgr,
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_stop")
        bridge.bind_run(r["run_id"], "test_session")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_called_once_with("run_stop")

    def test_stop_run_missing_runner_does_not_raise(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr, gateway_runner_ref=lambda: None)
        r = mgr.create_run("sess_stop")
        bridge.bind_run(r["run_id"], "test_session")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"

    def test_stop_run_already_terminal_no_interrupt(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_runner = MagicMock()
        mock_runner._running_agents = {"test_session": mock_agent}

        bridge = RuntimeControlBridge(
            mgr,
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_stop")
        bridge.bind_run(r["run_id"], "test_session")
        bridge.stop_run(r["run_id"])
        mock_agent.interrupt.assert_called_once()
        mock_agent.interrupt.reset_mock()

        result = bridge.stop_run(r["run_id"])
        assert result["terminal"] is True
        mock_agent.interrupt.assert_not_called()


class TestBridgeBindRun:
    """bind_run establishes run_id -> session_key mappings."""

    def test_bind_run_then_resolve_session_key(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "my_gw_key")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock:
            mock.return_value = 1
            bridge.resolve_approval(r["run_id"], "apr-001", "once")
            mock.assert_called_once_with("my_gw_key", "once")

    def test_callable_takes_priority_over_binding(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "callable_key",
        )
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "bound_key")

        # bind_run is checked first, not the callable
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock:
            mock.return_value = 1
            bridge.resolve_approval(r["run_id"], "apr-001", "once")
            mock.assert_called_once_with("bound_key", "once")

    def test_unbound_run_without_callable_no_live_primitive(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        mgr.request_approval(r["run_id"], "apr-001")
        with patch("tools.approval.resolve_gateway_approval") as mock:
            mock.return_value = 0
            result = bridge.resolve_approval(r["run_id"], "apr-001", "once")
            assert result["status"] == "resolved"
            mock.assert_not_called()

    def test_bind_run_stores_agent_reference(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key", mock_agent)
        assert bridge._live_agents.get(r["run_id"]) is mock_agent

    def test_bind_run_agent_is_optional(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key")
        assert r["run_id"] in bridge._bindings
        assert r["run_id"] not in bridge._live_agents

    def test_unbind_run_cleans_both(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "gw_key", mock_agent)
        bridge.unbind_run(r["run_id"])
        assert r["run_id"] not in bridge._bindings
        assert r["run_id"] not in bridge._live_agents

    def test_unbind_run_nonexistent_is_noop(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        bridge.unbind_run("no_such_run")


class TestBridgeStopDirectAgentRef:
    """stop_run uses direct agent reference from bind_run when available."""

    def test_stop_uses_direct_agent_not_gateway_ref(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_runner = MagicMock()
        mock_runner._running_agents = {"sk": MagicMock()}

        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "sk",
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk", mock_agent)
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_called_once_with("run_stop")
        mock_runner._running_agents["sk"].interrupt.assert_not_called()

    def test_stop_direct_agent_failure_does_not_raise(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_agent.interrupt.side_effect = RuntimeError("boom")
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk", mock_agent)
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"

    def test_stop_without_direct_agent_falls_back_to_gateway_ref(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        mock_runner = MagicMock()
        mock_runner._running_agents = {"sk": mock_agent}

        bridge = RuntimeControlBridge(
            mgr,
            get_session_key_for_run=lambda rid: "sk",
            gateway_runner_ref=lambda: mock_runner,
        )
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk")
        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_called_once_with("run_stop")

    def test_stop_after_unbind_uses_runmanager_only(self):
        mgr = RunManager()
        mock_agent = MagicMock()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run("sess_1")
        bridge.bind_run(r["run_id"], "sk", mock_agent)
        bridge.unbind_run(r["run_id"])

        result = bridge.stop_run(r["run_id"])
        assert result["status"] == "cancelled"
        mock_agent.interrupt.assert_not_called()
