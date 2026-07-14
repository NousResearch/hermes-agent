from __future__ import annotations

import importlib.util
import json
import sqlite3
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


def test_idempotency_key_is_unique_among_live_tasks(client: TestClient) -> None:
    """A duplicate live idempotency_key insert is rejected by the UNIQUE index.

    Simulates the TOCTOU race by inserting a second row directly (bypassing
    the SELECT-before-INSERT guard) — the partial UNIQUE index must refuse it.
    """
    created = _create(client, idempotency_key="race-key")
    task_id = created["task"]["id"]

    with kanban_db.connect(board="default") as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            with kanban_db.write_txn(conn):
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, status, created_at, workspace_kind, idempotency_key) "
                    "VALUES (?, ?, ?, ?, 'scratch', ?)",
                    ("t_duplicate", "dup", row["status"], row["created_at"], "race-key"),
                )

    # Archiving the live task frees the key: the partial index excludes
    # archived rows, so a fresh create with the same key is allowed (200/false
    # is only for a *live* duplicate; here a brand-new task is created).
    assert client.post(f"/api/plugins/kanban/tasks/{task_id}/archive").status_code == 200
    reused = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "after archive", "idempotency_key": "race-key"},
    )
    assert reused.status_code == 201, reused.text
    assert reused.json()["created"] is True
    assert reused.json()["task"]["id"] != task_id


def test_create_task_returns_existing_on_idempotency_race(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create deduped by the storage layer answers 200 / created=false.

    ``kanban_db.create_task_idempotent`` absorbs the UNIQUE-index race (see
    its tests in test_kanban_db.py) and reports ``created=False`` for the
    loser; the adapter must map that to HTTP 200 with the winner task, never
    201 or a 500.
    """
    original = _create(client, idempotency_key="winner-key")
    original_id = original["task"]["id"]

    monkeypatch.setattr(
        kanban_db,
        "create_task_idempotent",
        lambda conn, **kwargs: (original_id, False),
    )

    raced = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "raced", "idempotency_key": "winner-key"},
    )
    assert raced.status_code == 200, raced.text
    assert raced.json()["created"] is False
    assert raced.json()["task"]["id"] == original_id


def test_link_missing_task_is_404_not_400(client: TestClient) -> None:
    real_id = _create(client, idempotency_key="link-real")["task"]["id"]

    missing_parent = client.post(
        f"/api/plugins/kanban/tasks/t_nope/links/{real_id}"
    )
    assert missing_parent.status_code == 404, missing_parent.text

    missing_child = client.post(
        f"/api/plugins/kanban/tasks/{real_id}/links/t_nope"
    )
    assert missing_child.status_code == 404, missing_child.text

    # Unlink stays 404 for a missing endpoint (unchanged behaviour).
    missing_unlink = client.delete(
        f"/api/plugins/kanban/tasks/{real_id}/links/t_nope"
    )
    assert missing_unlink.status_code == 404, missing_unlink.text


def test_patch_rejects_edits_to_completed_task(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="done-edit")["task"]["id"]
    assert client.post(f"/api/plugins/kanban/tasks/{task_id}/complete").status_code == 200

    rejected = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}", json={"title": "too late"}
    )
    assert rejected.status_code == 409, rejected.text
    # The card text is unchanged.
    assert client.get(f"/api/plugins/kanban/tasks/{task_id}").json()["task"][
        "title"
    ] == "External operation"


def test_patch_rejects_edits_to_archived_task(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="archived-edit")["task"]["id"]
    assert client.post(f"/api/plugins/kanban/tasks/{task_id}/complete").status_code == 200
    assert client.post(f"/api/plugins/kanban/tasks/{task_id}/archive").status_code == 200

    rejected = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}", json={"body": "amend history"}
    )
    assert rejected.status_code == 409, rejected.text


def test_events_and_runs_limit_returns_most_recent_in_order(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="limit-key")["task"]["id"]

    with kanban_db.connect(board="default") as conn:
        with kanban_db.write_txn(conn):
            for i in range(5):
                conn.execute(
                    "INSERT INTO task_events "
                    "(task_id, run_id, kind, payload, created_at) "
                    "VALUES (?, NULL, ?, NULL, ?)",
                    (task_id, f"evt{i}", 2_000_000_000 + i),
                )
            for i in range(4):
                conn.execute(
                    "INSERT INTO task_runs "
                    "(task_id, status, started_at, ended_at, outcome) "
                    "VALUES (?, 'done', ?, ?, 'completed')",
                    (task_id, 2_000_000_000 + i, 2_000_000_000 + i + 1),
                )

    events = client.get(
        f"/api/plugins/kanban/tasks/{task_id}/events", params={"limit": 3}
    )
    assert events.status_code == 200
    kinds = [e["kind"] for e in events.json()["events"]]
    # The three newest events, oldest-first.
    assert kinds == ["evt2", "evt3", "evt4"]
    assert events.json()["count"] == 3

    runs = client.get(
        f"/api/plugins/kanban/tasks/{task_id}/runs", params={"limit": 2}
    )
    assert runs.status_code == 200
    starts = [r["started_at"] for r in runs.json()["runs"]]
    assert starts == [2_000_000_002, 2_000_000_003]
    assert runs.json()["count"] == 2


def test_error_detail_is_sanitized(client: TestClient) -> None:
    # An invalid board slug's raw ValueError text (regex description) must not
    # reach the client; a stable generic detail is returned instead.
    bad_board = client.get(
        "/api/plugins/kanban/tasks", params={"board": "Bad Slug!!"}
    )
    assert bad_board.status_code == 400, bad_board.text
    assert bad_board.json()["detail"] == "invalid board id"
    assert "1-64 chars" not in bad_board.text

    # A known-safe validation message is still surfaced verbatim (useful to
    # the caller, carries no internal detail).
    unknown_parent = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "orphan", "parents": ["t_missing"]},
    )
    assert unknown_parent.status_code == 400, unknown_parent.text
    assert "unknown parent task" in unknown_parent.json()["detail"]
