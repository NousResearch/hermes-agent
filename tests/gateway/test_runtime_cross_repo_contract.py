"""Cross-repo contract tests for Agent runtime <-> WebUI agent-runs adapter.

Verifies that Agent runtime response shapes match what the WebUI
agent-runs adapter expects.  These tests exercise the live contract:
- RunManager response shapes (create, status, events)
- Approval/clarify error mapping (not_found -> 404, conflict -> 409)
- Stop/cancel response shapes
- Secret redaction end-to-end
- The /v1/runs execution-plane status (control-plane-only)

No live HTTP or AIAgent is required -- RunManager is exercised directly
via its Python API, exactly as the WebUI adapter would see responses
after HTTP deserialisation.
"""

import json

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.models import redact_secrets


def _make_run_id() -> str:
    import uuid
    return f"run_{uuid.uuid4().hex}"


class TestCrossRepoRunCreation:
    """WebUI AgentRunsAdapter.start_run maps to RunManager.create_run.

    The WebUI adapter sends POST /v1/runs with session_id, message, etc.,
    and expects back: run_id, session_id, status.

    This test verifies RunManager.create_run produces the expected shape.
    """

    def test_create_run_shape_matches_webui_expectation(self):
        mgr = RunManager()
        body = {
            "session_id": "sess_webui_test",
            "message": "Hello from WebUI",
            "workspace": "/tmp",
            "profile": "default",
            "model": "claude-sonnet-4-5",
            "toolsets": ["file", "bash"],
            "metadata": {"client": "webui", "client_version": "unknown"},
        }
        result = mgr.create_run(
            session_id=body["session_id"],
            message=body["message"],
            workspace=body["workspace"],
            profile=body["profile"],
            model=body["model"],
            toolsets=body["toolsets"],
            metadata=body["metadata"],
        )
        assert result["run_id"].startswith("run_")
        assert result["session_id"] == "sess_webui_test"
        assert result["status"] == "queued"
        assert isinstance(result["controls"], list)

        serialized = json.dumps(result)
        assert "Bearer" not in serialized
        assert "api_key" not in serialized.lower() or "REDACTED" in serialized

    def test_create_run_minimal_body(self):
        mgr = RunManager()
        result = mgr.create_run(session_id="webui_min", message="hi")
        assert result["run_id"].startswith("run_")
        assert result["session_id"] == "webui_min"
        assert result["status"] == "queued"

    def test_create_run_returns_controls(self):
        mgr = RunManager()
        result = mgr.create_run(session_id="webui_ctrl")
        assert "observe" in result["controls"]
        assert "stop" in result["controls"]


class TestCrossRepoRunStatus:
    """WebUI AgentRunsAdapter.get_run maps to RunManager.get_status.

    The WebUI adapter expects: run_id, session_id, status, last_event_id,
    controls, terminal, pending_approval_id, pending_clarify_id.

    The Agent RunManager returns the superset of these fields.
    """

    def test_get_run_shape_matches_webui_expectation(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_stat")
        status = mgr.get_status(r["run_id"])

        assert status["run_id"] == r["run_id"]
        assert status["session_id"] == "webui_stat"
        assert status["status"] == "queued"
        assert "last_event_id" in status
        assert "last_seq" in status
        assert status["terminal"] is False
        assert isinstance(status["controls"], list)
        assert isinstance(status["pending_approval_ids"], list)
        assert isinstance(status["pending_clarify_ids"], list)

        serialized = json.dumps(status)
        assert "Bearer" not in serialized

    def test_get_run_unknown_returns_none(self):
        mgr = RunManager()
        assert mgr.get_status("nonexistent_run") is None


class TestCrossRepoEvents:
    """WebUI AgentRunsAdapter.observe_run maps to RunManager.read_events.

    The WebUI adapter expects: run_id, events array.  Each event needs:
    event_id, seq, type, created_at, terminal, payload.
    """

    def test_events_shape_matches_webui_expectation(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_evt")
        mgr.append_event(r["run_id"], "token.delta", payload={"text": "hello"})
        mgr.append_event(r["run_id"], "tool.started", payload={"tool": "bash"})

        result = mgr.read_events(r["run_id"])
        assert result["run_id"] == r["run_id"]
        assert len(result["events"]) == 3

        ev = result["events"][1]
        assert ev["type"] == "token.delta"
        assert ev["seq"] == 2
        assert ev["payload"]["text"] == "hello"
        assert "event_id" in ev
        assert "created_at" in ev
        assert "terminal" in ev

    def test_events_after_seq(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_seq")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        result = mgr.read_events(r["run_id"], after_seq=2)
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "event.b"

    def test_events_limit(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_lim")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        result = mgr.read_events(r["run_id"], limit=1)
        assert len(result["events"]) == 1

    def test_events_unknown_run_returns_none(self):
        mgr = RunManager()
        assert mgr.read_events("nonexistent") is None


class TestCrossRepoStop:
    """WebUI AgentRunsAdapter.cancel_run maps to RunManager.stop_run.

    The WebUI adapter expects a dict with ok/status/message fields.
    The underlying stop_run returns run_id, status, terminal, controls.
    """

    def test_stop_transitions_to_cancelled(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_stop")
        result = mgr.stop_run(r["run_id"])
        assert result["run_id"] == r["run_id"]
        assert result["status"] == "cancelled"
        assert result["terminal"] is True

    def test_stop_unknown_returns_not_found(self):
        mgr = RunManager()
        result = mgr.stop_run("nonexistent")
        assert result.get("error") == "not_found"


class TestCrossRepoApproval:
    """WebUI AgentRunsAdapter.respond_approval maps to RunManager.resolve_approval.

    Error mapping:
    - not_found -> 404 in WebUI (action_not_found)
    - conflict -> 409 in WebUI (already resolved)
    - success -> 200 with status=resolved
    """

    def test_approval_success_maps_to_resolved(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_appr")
        mgr.request_approval(r["run_id"], "appr_1")
        result = mgr.resolve_approval(r["run_id"], "appr_1", "approve")
        assert result["status"] == "resolved"
        assert result["type"] == "approval"

    def test_approval_not_found_maps_correctly(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_appr_nf")
        result = mgr.resolve_approval(r["run_id"], "appr_unknown", "approve")
        assert result["error"] == "not_found"
        assert "Approval" in result["message"]

    def test_approval_unknown_run_maps_correctly(self):
        mgr = RunManager()
        result = mgr.resolve_approval("run_nonexistent", "appr_1", "approve")
        assert result["error"] == "not_found"

    def test_approval_conflict_maps_correctly(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_appr_cf")
        mgr.request_approval(r["run_id"], "appr_1")
        mgr.resolve_approval(r["run_id"], "appr_1", "approve")
        result = mgr.resolve_approval(r["run_id"], "appr_1", "approve")
        assert result["error"] == "conflict"

    def test_approval_redacts_secrets(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_appr_redact")
        mgr.request_approval(r["run_id"], "appr_redact", payload={"api_key": "sk-secret-12345"})
        result = mgr.resolve_approval(r["run_id"], "appr_redact", "approve")
        serialized = json.dumps(result)
        assert "sk-secret-12345" not in serialized


class TestCrossRepoClarify:
    """WebUI AgentRunsAdapter.respond_clarify maps to RunManager.resolve_clarify.

    Error mapping:
    - not_found -> 404 in WebUI (action_not_found)
    - conflict -> 409 in WebUI (already resolved)
    - success -> 200 with status=resolved
    """

    def test_clarify_success_maps_to_resolved(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_clar")
        mgr.request_clarify(r["run_id"], "clar_1")
        result = mgr.resolve_clarify(r["run_id"], "clar_1", "yes")
        assert result["status"] == "resolved"
        assert result["type"] == "clarify"

    def test_clarify_not_found_maps_correctly(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_clar_nf")
        result = mgr.resolve_clarify(r["run_id"], "clar_unknown", "yes")
        assert result["error"] == "not_found"
        assert "Clarify" in result["message"]

    def test_clarify_unknown_run_maps_correctly(self):
        mgr = RunManager()
        result = mgr.resolve_clarify("run_nonexistent", "clar_1", "text")
        assert result["error"] == "not_found"

    def test_clarify_conflict_maps_correctly(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_clar_cf")
        mgr.request_clarify(r["run_id"], "clar_1")
        mgr.resolve_clarify(r["run_id"], "clar_1", "yes")
        result = mgr.resolve_clarify(r["run_id"], "clar_1", "yes")
        assert result["error"] == "conflict"

    def test_clarify_redacts_secrets(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_clar_redact")
        mgr.request_clarify(r["run_id"], "clar_redact", payload={"token": "sk-abc-12345"})
        result = mgr.resolve_clarify(r["run_id"], "clar_redact", "yes")
        serialized = json.dumps(result)
        assert "sk-abc-12345" not in serialized


class TestCrossRepoSecretRedaction:
    """Secret redaction must be preserved end-to-end across the contract."""

    def test_create_run_redacts_api_key_in_metadata(self):
        mgr = RunManager()
        r = mgr.create_run(
            session_id="webui_sec",
            metadata={"api_key": "sk-abc-12345"},
        )
        result = mgr.get_status(r["run_id"])
        serialized = json.dumps(result)
        assert "sk-abc-12345" not in serialized

    def test_events_redact_bearer_token(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="webui_sec_evt")
        mgr.append_event(r["run_id"], "error", payload={
            "message": "Auth failed",
            "auth": "Bearer sk-secret-99999",
        })
        result = mgr.read_events(r["run_id"])
        serialized = json.dumps(result)
        assert "sk-secret-99999" not in serialized

    def test_control_bridge_resolve_approval_redacts(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)
        r = mgr.create_run(session_id="webui_bridge_sec")
        mgr.request_approval(r["run_id"], "appr_sec", payload={
            "command": "echo secret",
            "api_key": "sk-bridge-secret",
        })
        result = bridge.resolve_approval(r["run_id"], "appr_sec", "approve")
        serialized = json.dumps(result)
        assert "sk-bridge-secret" not in serialized

    def test_redact_does_not_remove_nonsecret_data(self):
        mgr = RunManager()
        r = mgr.create_run(
            session_id="webui_nosec",
            metadata={"name": "test-user", "workspace": "/tmp"},
        )
        events = mgr.read_events(r["run_id"])
        serialized = json.dumps(events)
        assert "test-user" in serialized
        assert "/tmp" in serialized
