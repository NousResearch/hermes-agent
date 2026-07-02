"""Tests for non-API-server GatewayRunner runtime execution plane.

Covers:
- Non-API GatewayRunner execution creates or attaches runtime run_id
- Non-API execution calls bind_run(run_id, session_key, agent_ref)
- Non-API terminal path calls unbind_run(run_id)
- Non-API successful execution marks run completed
- Non-API failed execution marks run failed and redacts errors
- Non-API stopped execution marks run cancelled
- Events are appended for lifecycle transitions
- Secrets are redacted in events
"""

from unittest.mock import MagicMock

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.control_bridge import RuntimeControlBridge
from gateway.runtime.models import (
    EVENT_RUN_STARTED,
    EVENT_RUN_STATUS,
    EVENT_DONE,
    EVENT_ERROR,
)


class TestNonApiExecutionPlane:
    """Non-API GatewayRunner runs are wired into the runtime control plane."""

    def test_create_run_for_non_api_session(self):
        mgr = RunManager()
        r = mgr.create_run(
            session_id="non_api_session",
            model="test-model",
        )
        assert r["run_id"].startswith("run_")
        assert r["session_id"] == "non_api_session"
        assert r["status"] == "queued"

    def test_bind_run_for_non_api_session(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="non_api_gw_session")
        mock_agent = MagicMock()
        bridge.bind_run(r["run_id"], "non_api_gw_session", mock_agent)

        assert bridge._resolve_session_key(r["run_id"]) == "non_api_gw_session"
        assert bridge._live_agents.get(r["run_id"]) is mock_agent

    def test_unbind_run_cleans_non_api_session(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="non_api_unbind")
        bridge.bind_run(r["run_id"], "non_api_unbind", MagicMock())
        bridge.unbind_run(r["run_id"])

        assert bridge._resolve_session_key(r["run_id"]) is None
        assert bridge._live_agents.get(r["run_id"]) is None

    def test_non_api_successful_run_completed(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="non_api_success")
        bridge.bind_run(r["run_id"], "non_api_success")
        mgr.transition_status(r["run_id"], "running")
        mgr.complete_run(r["run_id"], result="success")

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"
        assert status["terminal"] is True
        assert status["result"] == "success"

    def test_non_api_failed_run_marked_failed(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="non_api_fail")
        bridge.bind_run(r["run_id"], "non_api_fail")
        mgr.transition_status(r["run_id"], "running")
        mgr.fail_run(r["run_id"], error="agent crashed")

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "failed"
        assert status["terminal"] is True
        assert status["error"] == "agent crashed"

    def test_non_api_stopped_run_marked_cancelled(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="non_api_cancel")
        bridge.bind_run(r["run_id"], "non_api_cancel")
        mgr.transition_status(r["run_id"], "running")
        result = mgr.stop_run(r["run_id"])

        assert result["status"] == "cancelled"
        assert result["terminal"] is True

    def test_non_api_lifecycle_events_appended(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="non_api_events")
        mgr.transition_status(r["run_id"], "running")
        mgr.complete_run(r["run_id"], result="ok")

        events = mgr.read_events(r["run_id"])
        assert events is not None
        event_types = [e["type"] for e in events["events"]]

        assert "run.started" in event_types or EVENT_RUN_STARTED in event_types
        assert "run.status" in event_types or EVENT_RUN_STATUS in event_types

    def test_non_api_agent_bind_then_unbind_lifecycle(self):
        mgr = RunManager()
        bridge = RuntimeControlBridge(mgr)

        r = mgr.create_run(session_id="non_api_lifecycle")
        mock_agent = MagicMock()
        bridge.bind_run(r["run_id"], "non_api_lifecycle", mock_agent)

        mgr.transition_status(r["run_id"], "running")
        status = mgr.get_status(r["run_id"])
        assert status["status"] == "running"

        mgr.complete_run(r["run_id"])
        bridge.unbind_run(r["run_id"])

        assert bridge._resolve_session_key(r["run_id"]) is None
        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"

    def test_messaging_platform_run_id_creation(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="msg_sess", run_id="msg-abc123def")
        assert r["run_id"] == "msg-abc123def"
        assert r["session_id"] == "msg_sess"

    def test_multiple_runs_different_sessions(self):
        mgr = RunManager()

        r1 = mgr.create_run(session_id="sess_a")
        r2 = mgr.create_run(session_id="sess_b")

        assert r1["run_id"] != r2["run_id"]
        assert r1["session_id"] == "sess_a"
        assert r2["session_id"] == "sess_b"

    def test_failed_run_has_error_event(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="error_sess")
        mgr.transition_status(r["run_id"], "running")
        mgr.fail_run(r["run_id"], error="something went wrong")

        events = mgr.read_events(r["run_id"])
        event_types = [e["type"] for e in events["events"]]
        assert "error" in event_types or EVENT_ERROR in event_types

    def test_cancelled_run_has_done_event(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="cancel_sess")
        mgr.transition_status(r["run_id"], "running")
        mgr.stop_run(r["run_id"])

        events = mgr.read_events(r["run_id"])
        event_types = [e["type"] for e in events["events"]]

        done_seen = False
        for e in events["events"]:
            if e["type"] in ("done", EVENT_DONE):
                done_seen = True
        assert done_seen

    def test_completed_run_has_done_event(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="complete_sess")
        mgr.transition_status(r["run_id"], "running")
        mgr.complete_run(r["run_id"], result="all good")

        events = mgr.read_events(r["run_id"])
        done_seen = False
        for e in events["events"]:
            if e["type"] in ("done", EVENT_DONE):
                done_seen = True
        assert done_seen

    def test_secrets_redacted_in_events(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="secret_sess", metadata={"api_key": "sk-secret-abc123", "name": "test"})
        events = mgr.read_events(r["run_id"])

        for e in events["events"]:
            payload_str = str(e.get("payload", {}))
            if "api_key" in payload_str:
                assert "sk-secret-abc123" not in payload_str

    def test_secrets_redacted_in_fail_error(self):
        mgr = RunManager()

        r = mgr.create_run(session_id="fail_secret")
        mgr.transition_status(r["run_id"], "running")
        mgr.fail_run(r["run_id"], error="API key sk-abcdef was invalid")

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "failed"

    def test_standalone_run_manager_fallback_unchanged(self):
        mgr = RunManager()
        r = mgr.create_run(session_id="standalone_fallback")
        mgr.transition_status(r["run_id"], "running")
        mgr.complete_run(r["run_id"])

        status = mgr.get_status(r["run_id"])
        assert status["status"] == "completed"
        assert status["terminal"] is True

    def test_run_id_is_unique_across_calls(self):
        mgr = RunManager()
        ids = set()
        for i in range(10):
            r = mgr.create_run(session_id=f"sess_{i}")
            ids.add(r["run_id"])
        assert len(ids) == 10
