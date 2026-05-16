"""Tests for BIF-494 protected cockpit dashboard endpoints."""

from __future__ import annotations

import json
import os
import pytest


SENSITIVE_KEYS = {
    "chat_id",
    "user_id",
    "thread_id",
    "session_key",
    "canonical_id",
    "arguments",
    "attachments",
    "codex_message_items",
    "codex_reasoning_items",
    "env",
    "files",
    "output",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
}
SENSITIVE_VALUES = {
    "C1234567890",
    "U1234567890",
    "T1234567890",
    "gateway-session-secret",
    "discord:C1234567890:U1234567890:T1234567890",
    "raw-secret-tool-arg",
    "raw-secret-tool-output",
    "raw-secret-env",
    "private chain of thought",
    "codex raw blob",
    "file contents should not leak",
    "sk-test-secret123",
    "https://api.internal.invalid/v1/runs",
    "https://api.internal.invalid/api/cockpit/events",
}


class FailOnWriteSessionDB:
    """Fake SessionDB that allows only read/list operations used by cockpit."""

    instances: list["FailOnWriteSessionDB"] = []

    def __init__(self):
        self.closed = False
        self.calls: list[tuple[str, dict]] = []
        FailOnWriteSessionDB.instances.append(self)

    def list_sessions_rich(self, *, limit: int = 20, offset: int = 0):
        self.calls.append(("list_sessions_rich", {"limit": limit, "offset": offset}))
        return [
            {
                "session_id": "discord:C1234567890:U1234567890:T1234567890",
                "source": "discord",
                "platform": "discord",
                "chat_type": "channel",
                "title": "Ops C1234567890",
                "status": "active",
                "updated_at": 123.0,
                "chat_id": "C1234567890",
                "user_id": "U1234567890",
                "thread_id": "T1234567890",
                "session_key": "gateway-session-secret",
                "canonical_id": "discord:C1234567890:U1234567890:T1234567890",
            },
            {
                "session_id": "cli-session-safe",
                "source": "cli",
                "platform": "cli",
                "chat_type": "operator",
                "title": "Operator",
                "status": "idle",
                "updated_at": 124.0,
            },
        ]

    def session_count(self):
        self.calls.append(("session_count", {}))
        return 2

    def _messages_for_session(self, session_id: str):
        assert session_id == "discord:C1234567890:U1234567890:T1234567890"
        return [
            {
                "id": 1,
                "session_id": session_id,
                "role": "user",
                "content": [
                    {"type": "text", "text": "please inspect https://api.internal.invalid/v1/runs for C1234567890 with API_KEY=raw-secret-env"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,file contents should not leak"}},
                ],
                "timestamp": 1710000000.0,
                "created_at": "2026-05-13T15:00:00Z",
                "status": "done",
                "attachments": [{"filename": "secret.txt", "content": "file contents should not leak"}],
            },
            {
                "id": 2,
                "session_id": session_id,
                "role": "assistant",
                "content": "I can help with sk-test-secret123 without exposing U1234567890.",
                "timestamp": 1710000001.0,
                "tool_calls": [{"function": {"name": "terminal", "arguments": "raw-secret-tool-arg"}}],
                "reasoning": "private chain of thought",
                "reasoning_content": "private chain of thought",
                "reasoning_details": [{"text": "private chain of thought"}],
                "codex_message_items": [{"raw": "codex raw blob"}],
                "codex_reasoning_items": [{"raw": "codex raw blob"}],
            },
            {
                "id": 3,
                "session_id": session_id,
                "role": "tool",
                "content": "raw-secret-tool-output",
                "tool_name": "terminal",
                "timestamp": 1710000002.0,
                "output": "raw-secret-tool-output",
            },
            {
                "id": 4,
                "session_id": session_id,
                "role": "user",
                "content": '[IMPORTANT: The user has invoked the "obsidian" skill, indicating they want you to follow its instructions. Full internal skill content here.',
                "timestamp": 1710000003.0,
            },
            {
                "id": 5,
                "session_id": session_id,
                "role": "user",
                "content": "[CONTEXT COMPACTION — REFERENCE ONLY] internal handoff summary",
                "timestamp": 1710000004.0,
            },
        ]

    def get_recent_messages(self, session_id: str, limit: int = 100):
        self.calls.append(("get_recent_messages", {"session_id": session_id, "limit": limit}))
        return self._messages_for_session(session_id)[-limit:]

    def get_messages(self, session_id: str):
        raise AssertionError("cockpit endpoint should use bounded get_recent_messages")

    def close(self):
        self.closed = True
        self.calls.append(("close", {}))

    def __getattr__(self, name: str):  # pragma: no cover - failure path only
        if any(word in name for word in ("add", "append", "create", "delete", "enqueue", "insert", "save", "set", "update", "write")):
            raise AssertionError(f"cockpit endpoint attempted write method {name}")
        raise AttributeError(name)


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - dependency-gated test suite
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli import web_server

    for name in tuple(os.environ):
        if name.startswith("HERMES_COCKPIT_SEND_"):
            monkeypatch.delenv(name, raising=False)
    web_server._COCKPIT_SEND_SERVICE = None
    web_server._COCKPIT_LANE_RESOLVER = None
    FailOnWriteSessionDB.instances.clear()
    monkeypatch.setattr(hermes_state, "SessionDB", FailOnWriteSessionDB)
    return TestClient(web_server.app)


@pytest.fixture()
def auth_headers():
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    return {_SESSION_HEADER_NAME: _SESSION_TOKEN}


def _assert_no_sensitive_content(value):
    if isinstance(value, dict):
        assert not (set(value) & SENSITIVE_KEYS)
        for nested in value.values():
            _assert_no_sensitive_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_sensitive_content(nested)
    elif isinstance(value, str):
        for sensitive in SENSITIVE_VALUES:
            assert sensitive not in value


def test_cockpit_capabilities_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/capabilities")

    assert response.status_code == 401


def test_cockpit_capabilities_returns_read_only_metadata_with_valid_token(client, auth_headers):
    response = client.get("/api/cockpit/capabilities", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["read_only"] is True
    assert body["input_enabled"] is True
    assert body["control_enabled"] is False
    assert body["external_send_enabled"] is False
    assert body["routing_enabled"] is False
    assert body["voice_enabled"] is False
    assert body["attachments_enabled"] is False
    assert body["local_chat"] == {
        "enabled": True,
        "transport": "dashboard_pty",
        "endpoint": "/api/pty",
        "scope": "local_dashboard_only",
    }
    assert "/api/cockpit/capabilities" in body["endpoints"]
    assert "/api/cockpit/lanes" in body["endpoints"]
    assert "/api/cockpit/n8n-checks" in body["endpoints"]
    assert "/api/cockpit/events" in body["endpoints"]
    assert "/api/cockpit/self-work-handoff" in body["endpoints"]
    handoff_endpoint = body["endpoints"]["/api/cockpit/self-work-handoff"]
    assert handoff_endpoint["methods"] == ["GET"]
    assert handoff_endpoint["actions_enabled"] is False
    assert handoff_endpoint["mutation_enabled"] is False
    assert body["transcript_window"] == {
        "kind": "recent",
        "bounded": True,
        "window_limit": 200,
        "total_scope": "bounded_recent_window",
    }
    messages_endpoint = body["endpoints"]["/api/cockpit/lanes/{lane_id}/messages"]
    assert messages_endpoint["window"] == body["transcript_window"]
    assert body["observer"]["enabled"] is True
    assert body["observer"]["read_only"] is True
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_lanes_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/lanes?limit=2&offset=1")

    assert response.status_code == 401
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_self_work_handoff_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/self-work-handoff")

    assert response.status_code == 401
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_self_work_handoff_returns_latest_resume_brief(client, auth_headers):
    from hermes_cli.self_work_handoff import write_self_work_handoff

    write_self_work_handoff(
        {
            "issue_identifier": "BIF-547",
            "goal": "Restart-safe self-work handoff",
            "current_phase": "verification",
            "last_action": "Added endpoint and tests",
            "next_safe_step": "Run focused pytest",
            "touched_files": ["hermes_cli/self_work_handoff.py", "hermes_cli/web_server.py"],
            "completed_verification": ["module normalization test drafted"],
            "pending_verification": ["endpoint auth test"],
            "operator_checklist": {
                "title": "BIF-547",
                "current_index": 1,
                "steps": [
                    {"label": "Persist handoff", "status": "done"},
                    {"label": "Verify endpoint", "status": "pending"},
                ],
            },
        }
    )

    response = client.get("/api/cockpit/self-work-handoff", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["read_only"] is True
    assert body["actions_enabled"] is False
    assert body["mutation_enabled"] is False
    assert body["has_handoff"] is True
    handoff = body["handoff"]
    assert handoff["issue_identifier"] == "BIF-547"
    assert handoff["current_phase"] == "verification"
    assert handoff["last_action"] == "Added endpoint and tests"
    assert "✓ done  Persist handoff" in handoff["rendered_checklist"]
    assert "▶ now   Verify endpoint" in handoff["rendered_checklist"]
    _assert_no_sensitive_content(body)
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_n8n_checks_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/n8n-checks")

    assert response.status_code == 401
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_quota_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/quota")

    assert response.status_code == 401
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_quota_returns_session_reset_recommendation(client, auth_headers):
    from hermes_state import get_hermes_home

    sessions_dir = get_hermes_home() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sessions.json").write_text(json.dumps({
        "expensive": {
            "session_id": "cockpit-expensive-session",
            "updated_at": "2026-05-14T12:00:00",
            "platform": "discord",
            "display_name": "Personal Assistant / #hermes",
            "last_prompt_tokens": 108865,
        }
    }))

    response = client.get("/api/cockpit/quota", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["read_only"] is True
    assert body["actions_enabled"] is False
    assert body["auto_reset_enabled"] is False
    assert body["max_prompt_tokens"] == 108865
    assert body["recommendation"]["level"] == "strong"
    assert body["recommendation"]["dedupe_key"] == "quota-session-reset-strong-100000"
    assert "fresh session" in body["recommendation"]["message"].lower()
    assert FailOnWriteSessionDB.instances == []


def test_public_status_quota_dedupe_key_is_display_safe(client):
    from hermes_state import get_hermes_home

    raw_session_id = "discord:C1234567890:U1234567890:T1234567890"
    sessions_dir = get_hermes_home() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sessions.json").write_text(json.dumps({
        "expensive": {
            "session_key": "discord:C1234567890:U1234567890:T1234567890",
            "session_id": raw_session_id,
            "created_at": "2026-05-14T10:00:00",
            "updated_at": "2026-05-14T12:00:00",
            "platform": "discord",
            "display_name": "Personal Assistant / #hermes",
            "last_prompt_tokens": 108865,
            "quota_warning_thresholds": [],
        }
    }))

    response = client.get("/api/status")

    assert response.status_code == 200
    rec = response.json()["session_quota_recommendation"]
    assert rec is not None
    assert rec["dedupe_key"].startswith("session-quota-public:100000:")
    assert raw_session_id not in rec["dedupe_key"]
    assert "C1234567890" not in rec["dedupe_key"]
    assert "U1234567890" not in rec["dedupe_key"]
    assert "T1234567890" not in rec["dedupe_key"]


def test_public_status_includes_active_biff_operating_mode(client, monkeypatch):
    monkeypatch.setenv("HERMES_BIFF_MODE", "emergency")

    response = client.get("/api/status")

    assert response.status_code == 200
    mode = response.json()["biff_operating_mode"]
    assert mode["name"] == "emergency"
    assert mode["label"] == "Emergency"
    assert mode["max_iterations"] == 16


def test_cockpit_n8n_checks_returns_read_only_inventory_with_live_or_fallback_source(client, auth_headers):
    response = client.get("/api/cockpit/n8n-checks", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["read_only"] is True
    assert body["source"] in {"n8n_live_latest_execution", "fixture_bif_525_inventory"}
    assert body["actions_enabled"] is False
    assert body["external_delivery_enabled"] is False
    assert body["generated_at"]
    assert isinstance(body["live"], bool)
    assert isinstance(body["fallback"], bool)
    assert isinstance(body["stale"], bool)
    assert "token" not in body.get("live_error", "").lower()
    assert "secret" not in body.get("live_error", "").lower()
    assert FailOnWriteSessionDB.instances == []

    expected_names = [
        "Morning Briefing",
        "Workflow Health Daily Report",
        "Auto-Remediation Monitor",
        "Immich Nightly Sync Monitor",
        "Obsidian Inbox Processor",
        "Alexa Bring Sync",
        "n8n Nightly Workflow Backup",
    ]
    assert [check["name"] for check in body["checks"]] == expected_names

    for check in body["checks"]:
        assert set(check) == {
            "id",
            "name",
            "status",
            "last_run",
            "next_schedule",
            "delivery",
            "auth",
            "action_needed",
            "summary",
            "live_source",
            "execution_status",
            "last_started",
            "last_completed",
            "output_summary",
        }
        assert check["auth"] in {"configured", "not_required", "unknown"}
        rendered = f"{check['summary']} {check['output_summary']} {check['delivery']}".lower()
        assert "token" not in rendered
        assert "secret" not in rendered
        assert "webhook" not in rendered
        assert "discord" not in check["delivery"].lower()


def test_cockpit_n8n_checks_endpoint_is_get_only(client, auth_headers):
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)("/api/cockpit/n8n-checks", headers=auth_headers)
        assert response.status_code == 405


@pytest.mark.parametrize(
    "path",
    ["/api/plugins/cockpit/capabilities", "/api/plugins/cockpit/lanes", "/api/plugins/cockpit/events"],
)
def test_cockpit_plugin_api_paths_are_not_public_spa_routes(client, path):
    response = client.get(path)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path", ["/api/does-not-exist", "/api"])
def test_unknown_api_paths_do_not_fall_through_to_spa(client, auth_headers, path):
    response = client.get(path, headers=auth_headers)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_cockpit_lanes_returns_display_safe_snapshots_and_reads_only(client, auth_headers):
    response = client.get("/api/cockpit/lanes?limit=50&offset=0", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["total"] == 2
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["lanes"]) == 2

    first, second = body["lanes"]
    assert first["schema_version"] == 1
    assert first["lane_id"].startswith("lane_")
    assert "session_id" not in first
    assert second["session_id"] == "cli-session-safe"
    assert second["alias"]["display_only"] is True
    _assert_no_sensitive_content(body)

    assert len(FailOnWriteSessionDB.instances) == 1
    db = FailOnWriteSessionDB.instances[0]
    assert db.closed is True
    assert db.calls == [
        ("list_sessions_rich", {"limit": 50, "offset": 0}),
        ("session_count", {}),
        ("session_count", {}),
        ("close", {}),
    ]


def test_cockpit_lane_messages_requires_dashboard_session_token(client):
    from hermes_cli.cockpit import build_lane_id

    lane_id = build_lane_id("discord", "discord:C1234567890:U1234567890:T1234567890")
    response = client.get(f"/api/cockpit/lanes/{lane_id}/messages?limit=100&offset=0")

    assert response.status_code == 401
    assert FailOnWriteSessionDB.instances == []


def _parse_sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data_lines = [line.removeprefix("data: ") for line in block.splitlines() if line.startswith("data: ")]
        if data_lines:
            events.append(json.loads("\n".join(data_lines)))
    return events


def test_cockpit_events_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/events?limit=1")

    assert response.status_code == 401
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_events_streams_text_event_stream_display_safe_events(client, auth_headers):
    from hermes_cli.cockpit import publish_cockpit_event

    publish_cockpit_event(
        {
            "type": "message.final",
            "platform": "discord",
            "canonical_id": "discord:C1234567890:U1234567890:T1234567890",
            "role": "assistant",
            "text": "done for C1234567890 at https://api.internal.invalid/api/cockpit/events with token=sk-test-secret123",
            "reasoning": "private chain of thought",
            "output": "raw-secret-tool-output",
        }
    )

    response = client.get("/api/cockpit/events?limit=2", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert len(events) == 2
    assert events[0]["type"] == "lane.status"
    assert events[1]["type"] == "lane.message.final"
    _assert_no_sensitive_content(events)
    assert "sk-test-secret123" not in json.dumps(events)
    assert FailOnWriteSessionDB.instances == []


def test_cockpit_lane_messages_unknown_lane_returns_404_and_reads_only(client, auth_headers):
    response = client.get("/api/cockpit/lanes/lane_unknown/messages?limit=100&offset=0", headers=auth_headers)

    assert response.status_code == 404
    assert len(FailOnWriteSessionDB.instances) == 1
    db = FailOnWriteSessionDB.instances[0]
    assert db.closed is True
    assert db.calls == [
        ("session_count", {}),
        ("list_sessions_rich", {"limit": 2, "offset": 0}),
        ("close", {}),
    ]


def test_cockpit_lane_messages_returns_display_safe_transcript_and_reads_only(client, auth_headers):
    from hermes_cli.cockpit import build_lane_id

    lane_id = build_lane_id("discord", "discord:C1234567890:U1234567890:T1234567890")
    response = client.get(f"/api/cockpit/lanes/{lane_id}/messages?limit=2&offset=1", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["lane_id"] == lane_id
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert body["total"] == 5
    assert body["bounded"] is True
    assert body["window_limit"] == 200
    assert body["total_scope"] == "bounded_recent_window"
    assert body["window"] == {
        "kind": "recent",
        "bounded": True,
        "window_limit": 200,
        "total_scope": "bounded_recent_window",
    }
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "assistant"
    assert body["messages"][0]["content"] == "I can help with [redacted] without exposing [id]."
    assert body["messages"][1]["role"] == "tool"
    assert body["messages"][1]["content"] == "[tool output redacted]"
    _assert_no_sensitive_content(body)

    hidden_response = client.get(f"/api/cockpit/lanes/{lane_id}/messages?limit=2&offset=3", headers=auth_headers)
    assert hidden_response.status_code == 200
    hidden_body = hidden_response.json()
    assert [message["content"] for message in hidden_body["messages"]] == [
        "[internal prompt hidden]",
        "[internal prompt hidden]",
    ]
    assert "obsidian" not in json.dumps(hidden_body)
    assert "CONTEXT COMPACTION" not in json.dumps(hidden_body)
    _assert_no_sensitive_content(hidden_body)

    assert len(FailOnWriteSessionDB.instances) == 2
    assert FailOnWriteSessionDB.instances[1].closed is True
    assert FailOnWriteSessionDB.instances[1].calls == [
        ("session_count", {}),
        ("list_sessions_rich", {"limit": 2, "offset": 0}),
        ("get_recent_messages", {"session_id": "discord:C1234567890:U1234567890:T1234567890", "limit": 200}),
        ("close", {}),
    ]

    db = FailOnWriteSessionDB.instances[0]
    assert db.closed is True
    assert db.calls == [
        ("session_count", {}),
        ("list_sessions_rich", {"limit": 2, "offset": 0}),
        ("get_recent_messages", {"session_id": "discord:C1234567890:U1234567890:T1234567890", "limit": 200}),
        ("close", {}),
    ]


def test_cockpit_lane_messages_uses_bounded_recent_reader_for_huge_session(monkeypatch, _isolate_hermes_home, auth_headers):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - dependency-gated test suite
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli import web_server
    from hermes_cli.cockpit import build_lane_id

    class HugeSessionDB(FailOnWriteSessionDB):
        instances: list["HugeSessionDB"] = []

        def __init__(self):
            super().__init__()
            HugeSessionDB.instances.append(self)

        def list_sessions_rich(self, *, limit: int = 20, offset: int = 0):
            self.calls.append(("list_sessions_rich", {"limit": limit, "offset": offset}))
            return [
                {
                    "session_id": "huge-safe-session",
                    "source": "cli",
                    "platform": "cli",
                    "chat_type": "operator",
                    "title": "Huge",
                    "updated_at": 2000.0,
                }
            ][offset : offset + limit]

        def session_count(self):
            self.calls.append(("session_count", {}))
            return 1

        def get_recent_messages(self, session_id: str, limit: int = 100):
            self.calls.append(("get_recent_messages", {"session_id": session_id, "limit": limit}))
            assert session_id == "huge-safe-session"
            assert limit == 200
            return [
                {"role": "assistant", "content": f"recent {index}", "timestamp": index}
                for index in range(997, 1000)
            ]

        def get_messages(self, session_id: str):  # pragma: no cover - must not be called
            raise AssertionError("huge full transcript should not be loaded")

    HugeSessionDB.instances.clear()
    monkeypatch.setattr(hermes_state, "SessionDB", HugeSessionDB)
    client = TestClient(web_server.app)

    lane_id = build_lane_id("cli", "huge-safe-session")
    response = client.get(f"/api/cockpit/lanes/{lane_id}/messages?limit=3", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["messages"] == [
        {"role": "assistant", "content": "recent 997", "timestamp": 997},
        {"role": "assistant", "content": "recent 998", "timestamp": 998},
        {"role": "assistant", "content": "recent 999", "timestamp": 999},
    ]
    assert body["total"] == 3
    assert body["bounded"] is True
    assert body["window_limit"] == 200
    assert body["total_scope"] == "bounded_recent_window"
    called = [name for name, _ in HugeSessionDB.instances[0].calls]
    assert "get_recent_messages" in called
    assert "get_messages" not in called


def test_cockpit_lane_messages_resolves_lanes_beyond_first_scan_page(monkeypatch, _isolate_hermes_home, auth_headers):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - dependency-gated test suite
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli import web_server
    from hermes_cli.cockpit import build_lane_id

    class ManySessionDB(FailOnWriteSessionDB):
        instances: list["ManySessionDB"] = []

        def __init__(self):
            super().__init__()
            ManySessionDB.instances.append(self)

        def list_sessions_rich(self, *, limit: int = 20, offset: int = 0):
            self.calls.append(("list_sessions_rich", {"limit": limit, "offset": offset}))
            sessions = [
                {
                    "session_id": f"cli-session-{index}",
                    "source": "cli",
                    "platform": "cli",
                    "chat_type": "operator",
                    "title": f"Operator {index}",
                }
                for index in range(201)
            ]
            return sessions[offset : offset + limit]

        def session_count(self):
            self.calls.append(("session_count", {}))
            return 201

        def get_recent_messages(self, session_id: str, limit: int = 100):
            self.calls.append(("get_recent_messages", {"session_id": session_id, "limit": limit}))
            assert session_id == "cli-session-200"
            return [{"role": "assistant", "content": "safe final", "timestamp": 1.0}]

        def get_messages(self, session_id: str):  # pragma: no cover - must not be called
            raise AssertionError("cockpit endpoint should use bounded get_recent_messages")

    ManySessionDB.instances.clear()
    monkeypatch.setattr(hermes_state, "SessionDB", ManySessionDB)
    client = TestClient(web_server.app)

    lane_id = build_lane_id("cli", "cli-session-200")
    response = client.get(f"/api/cockpit/lanes/{lane_id}/messages", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["messages"] == [{"role": "assistant", "content": "safe final", "timestamp": 1.0}]
    assert ManySessionDB.instances[0].calls == [
        ("session_count", {}),
        ("list_sessions_rich", {"limit": 200, "offset": 0}),
        ("list_sessions_rich", {"limit": 1, "offset": 200}),
        ("get_recent_messages", {"session_id": "cli-session-200", "limit": 200}),
        ("close", {}),
    ]


def test_cockpit_lane_messages_scan_is_bounded_to_1000_sessions(monkeypatch, _isolate_hermes_home, auth_headers):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - dependency-gated test suite
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli import web_server
    from hermes_cli.cockpit import build_lane_id

    class OverBoundSessionDB(FailOnWriteSessionDB):
        instances: list["OverBoundSessionDB"] = []

        def __init__(self):
            super().__init__()
            OverBoundSessionDB.instances.append(self)

        def list_sessions_rich(self, *, limit: int = 20, offset: int = 0):
            self.calls.append(("list_sessions_rich", {"limit": limit, "offset": offset}))
            return [
                {
                    "session_id": f"cli-session-{index}",
                    "source": "cli",
                    "platform": "cli",
                    "chat_type": "operator",
                    "title": f"Operator {index}",
                }
                for index in range(offset, min(offset + limit, 1001))
            ]

        def session_count(self):
            self.calls.append(("session_count", {}))
            return 1001

        def get_recent_messages(self, session_id: str, limit: int = 100):
            raise AssertionError("out-of-bound lane should not read messages")

        def get_messages(self, session_id: str):  # pragma: no cover - must not be called
            raise AssertionError("out-of-bound lane should not read messages")

    OverBoundSessionDB.instances.clear()
    monkeypatch.setattr(hermes_state, "SessionDB", OverBoundSessionDB)
    client = TestClient(web_server.app)

    lane_id = build_lane_id("cli", "cli-session-1000")
    response = client.get(f"/api/cockpit/lanes/{lane_id}/messages", headers=auth_headers)

    assert response.status_code == 404
    assert OverBoundSessionDB.instances[0].calls == [
        ("session_count", {}),
        ("list_sessions_rich", {"limit": 200, "offset": 0}),
        ("list_sessions_rich", {"limit": 200, "offset": 200}),
        ("list_sessions_rich", {"limit": 200, "offset": 400}),
        ("list_sessions_rich", {"limit": 200, "offset": 600}),
        ("list_sessions_rich", {"limit": 200, "offset": 800}),
        ("close", {}),
    ]


def test_cockpit_lane_messages_logical_lane_merges_only_25_mapped_sessions(monkeypatch, _isolate_hermes_home, auth_headers):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - dependency-gated test suite
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli import web_server
    from hermes_cli.cockpit import build_lane_id

    raw_session_ids = {f"operator-secret:{index}:U1234567890" for index in range(30)}

    class LogicalLaneSessionDB(FailOnWriteSessionDB):
        instances: list["LogicalLaneSessionDB"] = []

        def __init__(self):
            super().__init__()
            LogicalLaneSessionDB.instances.append(self)

        def list_sessions_rich(self, *, limit: int = 20, offset: int = 0):
            self.calls.append(("list_sessions_rich", {"limit": limit, "offset": offset}))
            return [
                {
                    "session_id": f"operator-secret:{index}:U1234567890",
                    "source": "cli",
                    "platform": "cli",
                    "chat_type": "operator",
                    "role": "biff",
                    "title": "Biff",
                    "updated_at": float(2000 - index),
                }
                for index in range(offset, min(offset + limit, 30))
            ]

        def session_count(self):
            self.calls.append(("session_count", {}))
            return 30

        def get_recent_messages(self, session_id: str, limit: int = 100):
            self.calls.append(("get_recent_messages", {"session_id": session_id, "limit": limit}))
            index = int(session_id.split(":")[1])
            return [{"role": "assistant", "content": f"safe logical message {index}", "timestamp": f"{index:03d}"}]

        def get_messages(self, session_id: str):  # pragma: no cover - must not be called
            raise AssertionError("cockpit endpoint should use bounded get_recent_messages")

    LogicalLaneSessionDB.instances.clear()
    monkeypatch.setattr(hermes_state, "SessionDB", LogicalLaneSessionDB)
    client = TestClient(web_server.app)

    lane_id = build_lane_id("cockpit.logical", "operator/current")
    response = client.get(f"/api/cockpit/lanes/{lane_id}/messages?limit=5&offset=10", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["lane_id"] == lane_id
    assert body["limit"] == 5
    assert body["offset"] == 10
    assert body["total"] == 25
    assert [message["content"] for message in body["messages"]] == [
        "safe logical message 10",
        "safe logical message 11",
        "safe logical message 12",
        "safe logical message 13",
        "safe logical message 14",
    ]
    body_json = json.dumps(body)
    assert not any(raw_id in body_json for raw_id in raw_session_ids)
    _assert_no_sensitive_content(body)

    calls = LogicalLaneSessionDB.instances[0].calls
    assert calls[:2] == [
        ("session_count", {}),
        ("list_sessions_rich", {"limit": 30, "offset": 0}),
    ]
    get_message_calls = [call for call in calls if call[0] == "get_recent_messages"]
    assert len(get_message_calls) == 25
    assert all(call[1]["limit"] == 200 for call in get_message_calls)
    assert calls[-1] == ("close", {})


@pytest.mark.parametrize("path", ["/api/cockpit/capabilities", "/api/cockpit/lanes", "/api/cockpit/events", "/api/cockpit/lanes/lane_x/messages"])
def test_cockpit_endpoints_are_get_only(client, auth_headers, path):
    response = client.post(path, headers=auth_headers)

    assert response.status_code == 405
