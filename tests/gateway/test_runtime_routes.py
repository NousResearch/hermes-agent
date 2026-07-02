"""API contract tests for /v1/runs runtime routes.

Validates the expected request/response shapes that route handlers
must conform to.  The route handlers live in api_server.py (aiohttp);
these tests exercise the RunManager through the contract shapes that
HTTP endpoints would produce and consume.

Also validates that responses redact secrets and that error shapes
match the project's standard error format.
"""

import pytest

from gateway.runtime.run_manager import RunManager
from gateway.runtime.models import redact_secrets


def _make_request_body(**overrides):
    base = {
        "session_id": "sess_123",
        "message": "User request",
        "workspace": "/home/user/workspace",
        "profile": "default",
        "model": "provider/model",
        "toolsets": ["terminal", "file"],
        "metadata": {"client": "webui", "client_version": "unknown"},
    }
    base.update(overrides)
    return base


class TestCreateRunContract:
    def test_post_v1_runs_returns_run_id_status_urls(self):
        mgr = RunManager()
        body = _make_request_body()
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
        assert result["session_id"] == "sess_123"
        assert result["status"] == "queued"
        assert result["events_url"] == f"/v1/runs/{result['run_id']}/events"
        assert result["status_url"] == f"/v1/runs/{result['run_id']}"
        assert "observe" in result["controls"]
        assert "stop" in result["controls"]

    def test_creates_with_minimal_body(self):
        mgr = RunManager()
        result = mgr.create_run(session_id="sess_min")
        assert result["run_id"].startswith("run_")
        assert result["status"] == "queued"

    def test_response_redacts_secrets_in_payload(self):
        mgr = RunManager()
        body = _make_request_body()
        body["metadata"]["api_key"] = "sk-secret-key-12345"
        result = mgr.create_run(
            session_id=body["session_id"],
            message=body["message"],
            metadata=body["metadata"],
        )
        events = mgr.read_events(result["run_id"])
        started = events["events"][0]
        meta = started["payload"]["metadata"]
        assert meta.get("api_key") != "sk-secret-key-12345"


class TestGetRunContract:
    def test_get_v1_runs_run_id_returns_runtime_status(self):
        mgr = RunManager()
        r = mgr.create_run("sess_123")
        status = mgr.get_status(r["run_id"])
        assert status is not None
        assert status["run_id"] == r["run_id"]
        assert status["session_id"] == "sess_123"
        assert status["status"] == "queued"
        assert "last_event_id" in status
        assert "last_seq" in status
        assert "terminal" in status
        assert "controls" in status
        assert "pending_approval_ids" in status
        assert "pending_clarify_ids" in status
        assert "error" in status
        assert "result" in status

    def test_unknown_run_returns_none(self):
        mgr = RunManager()
        assert mgr.get_status("run_nonexistent") is None

    def test_status_redacts_secrets(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        status = mgr.get_status(r["run_id"])
        status["error"] = "failed with api_key=sk-proj-verylongsecretkeythatmatchespattern"
        redacted = redact_secrets(status)
        assert "sk-proj-verylongsecretkeythatmatchespattern" not in str(redacted)


class TestGetEventsContract:
    def test_get_v1_runs_run_id_events_returns_replay_payload(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.append_event(r["run_id"], "tool.started")
        mgr.append_event(r["run_id"], "tool.done")
        result = mgr.read_events(r["run_id"])
        assert result is not None
        assert result["run_id"] == r["run_id"]
        assert "events" in result
        assert len(result["events"]) == 3

    def test_events_support_after_seq(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        result = mgr.read_events(r["run_id"], after_seq=2)
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "event.b"

    def test_events_support_limit(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.append_event(r["run_id"], "event.a")
        mgr.append_event(r["run_id"], "event.b")
        result = mgr.read_events(r["run_id"], limit=1)
        assert len(result["events"]) == 1

    def test_unknown_run_returns_none(self):
        mgr = RunManager()
        assert mgr.read_events("run_nonexistent") is None

    def test_events_redact_secrets(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.append_event(r["run_id"], "error", payload={
            "message": "Connection failed",
            "auth": "Bearer sk-secret-12345",
        })
        result = mgr.read_events(r["run_id"])
        error_event = result["events"][1]
        assert error_event["payload"].get("auth") != "Bearer sk-secret-12345"


class TestStopRunContract:
    def test_post_stop_returns_status_transition(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.stop_run(r["run_id"])
        assert result["run_id"] == r["run_id"]
        assert result["status"] == "cancelled"
        assert result["terminal"] is True

    def test_unknown_run_returns_not_found(self):
        mgr = RunManager()
        result = mgr.stop_run("run_nonexistent")
        assert result.get("error") == "not_found"


class TestApprovalContract:
    def test_post_approval_unknown_id_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.resolve_approval(r["run_id"], "apr_unknown", "approve")
        assert result["error"] == "not_found"
        assert "message" in result

    def test_approval_unknown_run_not_found(self):
        mgr = RunManager()
        result = mgr.resolve_approval("run_nonexistent", "apr-1", "deny")
        assert result["error"] == "not_found"

    def test_approval_resolution_works_with_request(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_approval(r["run_id"], "apr-1")
        result = mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        assert result["status"] == "resolved"
        assert result["type"] == "approval"


class TestClarifyContract:
    def test_post_clarify_unknown_id_returns_not_found(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        result = mgr.resolve_clarify(r["run_id"], "clar_unknown", "yes")
        assert result["error"] == "not_found"
        assert "message" in result

    def test_clarify_unknown_run_not_found(self):
        mgr = RunManager()
        result = mgr.resolve_clarify("run_nonexistent", "clar-1", "text")
        assert result["error"] == "not_found"

    def test_clarify_resolution_works_with_request(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_clarify(r["run_id"], "clar-1")
        result = mgr.resolve_clarify(r["run_id"], "clar-1", "yes")
        assert result["status"] == "resolved"
        assert result["type"] == "clarify"


class TestErrorShapes:
    def test_not_found_error_has_standard_shape(self):
        mgr = RunManager()
        result = mgr.stop_run("run_nonexistent")
        assert "error" in result
        assert "message" in result
        assert result["error"] == "not_found"
        assert isinstance(result["message"], str)

    def test_conflict_error_has_standard_shape(self):
        mgr = RunManager()
        r = mgr.create_run("sess")
        mgr.request_approval(r["run_id"], "apr-1")
        mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        result = mgr.resolve_approval(r["run_id"], "apr-1", "approve")
        assert "error" in result
        assert "message" in result
        assert result["error"] == "conflict"

    def test_redaction_applied_to_error_responses(self):
        mgr = RunManager()
        r = mgr.create_run("sess", metadata={"token": "sk-abc-secret-key"})
        events = mgr.read_events(r["run_id"])
        serialized = str(events)
        assert "sk-abc-secret-key" not in serialized


class TestMalformedRequests:
    def test_unknown_run_deterministic(self):
        mgr = RunManager()
        ids = ["", "run_fake", "nonexistent", " "]
        for rid in ids:
            assert mgr.get_status(rid) is None
            assert mgr.read_events(rid) is None
            assert mgr.stop_run(rid).get("error") == "not_found"
            assert mgr.resolve_approval(rid, "apr-1", "approve").get("error") == "not_found"
            assert mgr.resolve_clarify(rid, "clar-1", "text").get("error") == "not_found"
