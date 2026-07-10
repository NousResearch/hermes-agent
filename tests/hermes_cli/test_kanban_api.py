from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db
from hermes_cli.kanban_api import router


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kanban_db._INITIALIZED_PATHS.clear()
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/kanban")
    with TestClient(app) as test_client:
        yield test_client


def _create(client: TestClient, **overrides) -> dict:
    payload = {
        "title": "External operation",
        "body": "private execution instructions",
        "tenant": "ops",
        "priority": 5,
        "idempotency_key": "operation-001",
    }
    payload.update(overrides)
    response = client.post("/api/plugins/kanban/tasks", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_health_capabilities_and_safe_board_dtos(client: TestClient) -> None:
    health = client.get("/api/plugins/kanban/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "hermes-kanban",
        "api_version": "1",
        "current_board": "default",
    }

    capabilities = client.get("/api/plugins/kanban/capabilities")
    assert capabilities.status_code == 200
    body = capabilities.json()
    assert body["idempotent_task_creation"] is True
    assert body["profile_execution"] is False
    assert "ready" in body["task_statuses"]

    boards = client.get("/api/plugins/kanban/boards").json()
    assert boards["current"] == "default"
    assert boards["boards"][0]["id"] == "default"
    assert "db_path" not in boards["boards"][0]
    assert "default_workdir" not in boards["boards"][0]

    detail = client.get("/api/plugins/kanban/boards/Default")
    assert detail.status_code == 200
    assert detail.json()["board"]["id"] == "default"


def test_create_list_get_patch_are_idempotent_and_sanitized(client: TestClient) -> None:
    first = _create(client)
    task_id = first["task"]["id"]
    assert first["created"] is True
    assert first["task"]["links"] == {"parents": [], "children": []}

    repeated = client.post(
        "/api/plugins/kanban/tasks",
        headers={"Idempotency-Key": "operation-001"},
        json={"title": "ignored on replay"},
    )
    assert repeated.status_code == 200
    assert repeated.json()["created"] is False
    assert repeated.json()["task"]["id"] == task_id

    listed = client.get("/api/plugins/kanban/tasks", params={"tenant": "ops"})
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    detail = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert detail.status_code == 200
    task = detail.json()["task"]
    assert task["title"] == "External operation"
    forbidden = {
        "body", "result", "workspace_path", "branch_name", "claim_lock",
        "worker_pid", "session_id", "idempotency_key", "last_failure_error",
    }
    assert forbidden.isdisjoint(task)
    assert "private execution instructions" not in detail.text

    patched = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"title": "Renamed operation", "priority": 9, "assignee": "Worker-One"},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["task"]["title"] == "Renamed operation"
    assert patched.json()["task"]["priority"] == 9
    assert patched.json()["task"]["assignee"] == "worker-one"


def test_links_actions_and_observability_are_sanitized(client: TestClient) -> None:
    parent_id = _create(client, title="Parent", idempotency_key="parent-001")["task"]["id"]
    child_id = _create(client, title="Child", idempotency_key="child-001")["task"]["id"]

    linked = client.post(f"/api/plugins/kanban/tasks/{parent_id}/links/{child_id}")
    assert linked.status_code == 200, linked.text
    child = client.get(f"/api/plugins/kanban/tasks/{child_id}").json()["task"]
    assert child["links"]["parents"] == [parent_id]
    assert child["status"] == "todo"

    comment = client.post(
        f"/api/plugins/kanban/tasks/{parent_id}/comment",
        json={"body": "private operator note", "author": "ops-dashboard"},
    )
    assert comment.status_code == 201
    assert "private operator note" not in comment.text

    completed = client.post(
        f"/api/plugins/kanban/tasks/{parent_id}/complete",
        json={"summary": "private completion handoff"},
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["task"]["status"] == "done"
    assert "private completion handoff" not in completed.text

    events = client.get(f"/api/plugins/kanban/tasks/{parent_id}/events")
    assert events.status_code == 200
    assert events.json()["events"]
    assert all("payload" not in event for event in events.json()["events"])
    assert "private" not in events.text

    with kanban_db.connect(board="default") as conn:
        now = 1_700_000_000
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs "
                "(task_id, profile, status, started_at, ended_at, outcome, summary, metadata, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (parent_id, "private-profile", "done", now, now + 1, "completed",
                 "raw private summary", json.dumps({"workspace": "/srv/private/work"}),
                 "secret error"),
            )

    runs = client.get(f"/api/plugins/kanban/tasks/{parent_id}/runs")
    assert runs.status_code == 200
    run = runs.json()["runs"][-1]
    assert set(run) == {"id", "status", "outcome", "started_at", "ended_at"}
    assert "private-profile" not in runs.text
    assert "/srv/private/work" not in runs.text

    log_path = kanban_db.worker_log_path(parent_id, board="default")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "working in /srv/private/worktree\nAuthorization: Bearer secret-value-1234567890\n",
        encoding="utf-8",
    )
    log_response = client.get(f"/api/plugins/kanban/tasks/{parent_id}/log")
    assert log_response.status_code == 200
    log_body = log_response.json()
    assert "path" not in log_body
    assert "/srv/private" not in log_body["excerpt"]
    assert "secret-value-1234567890" not in log_body["excerpt"]

    unlinked = client.delete(f"/api/plugins/kanban/tasks/{parent_id}/links/{child_id}")
    assert unlinked.status_code == 200
    assert unlinked.json()["removed"] is True

    blocked = client.post(
        f"/api/plugins/kanban/tasks/{child_id}/block",
        json={"reason": "private blocker", "kind": "needs_input"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["task"]["status"] == "blocked"
    assert "private blocker" not in blocked.text

    assert client.post(f"/api/plugins/kanban/tasks/{child_id}/unblock").status_code == 200
    assert client.post(f"/api/plugins/kanban/tasks/{child_id}/archive").status_code == 200


def test_dashboard_legacy_routes_are_namespaced_after_public_router() -> None:
    plugin_path = Path(__file__).parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    module_name = "test_hermes_dashboard_plugin_kanban"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        paths = {getattr(route, "path", "") for route in module.router.routes}
    finally:
        sys.modules.pop(module_name, None)

    assert "/health" in paths
    assert "/tasks" in paths
    assert "/dashboard/board" in paths
    assert "/dashboard/tasks/{task_id}" in paths
