"""Tests for GET /api/fleet/activity (hermes_cli.web_routers.fleet).

Real imports against a temp HERMES_HOME (no mocks): a real kanban board DB with a
running task, and a real state.db gateway-routing-index row with an in-flight turn.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def _client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _create_running_kanban_task(*, board=None, title="fleet test task", assignee="anika"):
    """Create a task and claim it (``ready`` -> ``running``) the way the real dispatcher
    does. ``create_task(initial_status="running")`` does NOT itself produce a running row —
    ``initial_task_state`` only ever resolves to blocked/triage/todo/ready — running is only
    reached through ``claim_task``'s atomic CAS, which is also what stamps ``started_at``."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    conn = connect(board=board)
    try:
        task_id = kb.create_task(conn, title=title, assignee=assignee, board=board)
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None, "claim_task should succeed on a freshly created ready task"
    finally:
        conn.close()
    return task_id


def _write_gateway_routing_entry(session_key: str, entry: dict):
    from hermes_state import SessionDB, _default_db_path

    db = SessionDB(db_path=_default_db_path())
    try:
        db.save_gateway_routing_entry(session_key, json.dumps(entry))
    finally:
        db.close()


class TestFleetActivityEndpoint:
    def test_empty_install_returns_empty_lists(self, _client):
        resp = _client.get("/api/fleet/activity")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"kanban_tasks": [], "gateway_sessions": [], "count": 0}

    def test_running_kanban_task_is_surfaced_with_elapsed_and_heartbeat(self, _client):
        task_id = _create_running_kanban_task(title="ship the fleet panel", assignee="anika")

        resp = _client.get("/api/fleet/activity")
        assert resp.status_code == 200
        body = resp.json()

        assert body["count"] == 1
        assert len(body["kanban_tasks"]) == 1
        row = body["kanban_tasks"][0]
        assert row["task_id"] == task_id
        assert row["title"] == "ship the fleet panel"
        assert row["profile"] == "anika"
        assert row["board"] == "default"
        assert isinstance(row["started_at"], (int, float))
        assert row["elapsed_seconds"] >= 0
        # Freshly created task has no heartbeat yet.
        assert row["last_heartbeat_at"] is None
        assert row["heartbeat_age_seconds"] is None
        assert body["gateway_sessions"] == []

    def test_non_running_kanban_tasks_are_excluded(self, _client):
        from hermes_cli import kanban_db as kb
        from hermes_cli.kanban_db_connect import connect

        conn = connect()
        try:
            kb.create_task(conn, title="still queued", assignee="anika")
        finally:
            conn.close()

        resp = _client.get("/api/fleet/activity")
        assert resp.status_code == 200
        assert resp.json()["kanban_tasks"] == []

    def test_kanban_tasks_across_named_boards_are_aggregated(self, _client):
        from hermes_cli import kanban_db as kb

        kb.create_board("ops")
        _create_running_kanban_task(board=None, title="default board task", assignee="anika")
        _create_running_kanban_task(board="ops", title="ops board task", assignee="raaj")

        resp = _client.get("/api/fleet/activity")
        assert resp.status_code == 200
        rows = resp.json()["kanban_tasks"]
        boards = {row["board"] for row in rows}
        assert boards == {"default", "ops"}
        assert {row["title"] for row in rows} == {"default board task", "ops board task"}

    def test_gateway_session_with_active_turn_is_surfaced(self, _client):
        from hermes_cli.profiles import get_active_profile_name

        started = datetime.now(timezone.utc).isoformat()
        _write_gateway_routing_entry("telegram:12345", {
            "session_key": "telegram:12345",
            "session_id": "sess-abc",
            "created_at": started,
            "updated_at": started,
            "platform": "telegram",
            "display_name": "Raaj",
            "chat_type": "dm",
            "active_turn_token": "tok-1",
            "active_turn_started_at": started,
        })

        resp = _client.get("/api/fleet/activity")
        assert resp.status_code == 200
        body = resp.json()

        assert body["kanban_tasks"] == []
        assert len(body["gateway_sessions"]) == 1
        row = body["gateway_sessions"][0]
        assert row["session_key"] == "telegram:12345"
        assert row["platform"] == "telegram"
        assert row["display_name"] == "Raaj"
        assert row["profile"] == get_active_profile_name()
        assert row["elapsed_seconds"] >= 0
        assert body["count"] == 1

    def test_gateway_session_without_active_turn_is_excluded(self, _client):
        _write_gateway_routing_entry("telegram:99999", {
            "session_key": "telegram:99999",
            "session_id": "sess-idle",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "platform": "telegram",
            "display_name": "Idle User",
            "chat_type": "dm",
            "active_turn_token": None,
            "active_turn_started_at": None,
        })

        resp = _client.get("/api/fleet/activity")
        assert resp.status_code == 200
        assert resp.json()["gateway_sessions"] == []
