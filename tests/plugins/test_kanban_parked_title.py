from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    plugin_file = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("parked_title_plugin", plugin_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _create(title: str, *, initial_status: str = "running") -> str:
    with kbc.connect() as conn:
        return kb.create_task(conn, title=title, initial_status=initial_status)


def _column(client: TestClient, status: str) -> list[dict]:
    columns = client.get("/api/plugins/kanban/board").json()["columns"]
    return next(column["tasks"] for column in columns if column["name"] == status)


def test_explicit_hold_projects_label_on_board_and_detail(client):
    task_id = _create("[research] Evaluate options", initial_status="blocked")

    card = next(task for task in _column(client, "blocked") if task["id"] == task_id)
    detail = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()["task"]
    assert card["title"] == detail["title"] == "[research] Evaluate options"
    assert card["display_title"] == detail["display_title"] == "[parked] Evaluate options"

    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).title == "[research] Evaluate options"


def test_title_edits_during_hold_remain_raw_and_project_live(client):
    task_id = _create("Before", initial_status="blocked")
    response = client.patch(f"/api/plugins/kanban/tasks/{task_id}", json={"title": "[planning] Revised"})

    assert response.status_code == 200
    assert response.json()["task"]["title"] == "[planning] Revised"
    assert response.json()["task"]["display_title"] == "[parked] Revised"
    assert next(t for t in _column(client, "blocked") if t["id"] == task_id)["display_title"] == "[parked] Revised"


def test_unblock_restores_title_and_later_reblock_does_not_reuse_hold(client):
    task_id = _create("Plain title", initial_status="blocked")
    with kbc.connect() as conn:
        assert kb.unblock_task(conn, task_id)
    restored = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()["task"]
    assert restored["display_title"] == restored["title"] == "Plain title"

    with kbc.connect() as conn:
        assert kb.block_task(conn, task_id, reason="retry failed", kind="transient")
    reblocked = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()["task"]
    assert reblocked["status"] == "blocked"
    assert reblocked["display_title"] == "Plain title"


@pytest.mark.parametrize("reason", ["failure", "dependency", "approval"])
def test_other_block_reasons_do_not_project_parked_label(client, reason):
    task_id = _create("[research] Ordinary block")
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task_id,))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'blocked', ?, ?)",
            (task_id, json.dumps({"reason": reason}), int(time.time())),
        )

    task = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()["task"]
    assert task["display_title"] == "[research] Ordinary block"


def test_unblocked_creation_parameter_does_not_mean_initial_hold(client):
    task_id = _create("[research] Automatic start")
    task = client.get(f"/api/plugins/kanban/tasks/{task_id}").json()["task"]
    assert task["status"] == "ready"
    assert task["display_title"] == "[research] Automatic start"
