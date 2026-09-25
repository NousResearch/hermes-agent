"""Kanban dashboard plugin: the board's done column is completion history.

``GET /board`` buckets one ``list_tasks()`` fetch, so every column inherits
the shared ``priority DESC, created_at ASC`` order — which says nothing
about when finished work finished. These pin the done column to
newest-completed-first while a queue lane keeps the FIFO default.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_done_order_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._dashboard_router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _column(client, name):
    board = client.get("/api/plugins/kanban/board").json()
    return next(c["tasks"] for c in board["columns"] if c["name"] == name)


def _finish(client, task_id):
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "done", "result": "done", "summary": "done"},
    )
    assert r.status_code == 200, r.text


def test_done_column_orders_by_completion_time_not_creation(client):
    first = client.post("/api/plugins/kanban/tasks", json={"title": "created first"}).json()["task"]
    second = client.post("/api/plugins/kanban/tasks", json={"title": "created second"}).json()["task"]
    _finish(client, first["id"])
    _finish(client, second["id"])

    # complete_task stamps one wall clock for both, so pin distinct
    # timestamps: the card created first also finished last.
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET completed_at = 2000 WHERE id = ?", (first["id"],))
        conn.execute("UPDATE tasks SET completed_at = 1000 WHERE id = ?", (second["id"],))
        conn.commit()

    done = _column(client, "done")
    assert [t["id"] for t in done] == [first["id"], second["id"]]


def test_queue_lane_keeps_fifo_default(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    # Same-second creations tie on the shared order; pin explicit ages.
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET created_at = 200 WHERE id = ?", (a["id"],))
        conn.execute("UPDATE tasks SET created_at = 100 WHERE id = ?", (b["id"],))
        conn.commit()

    ready = _column(client, "ready")  # no parents -> both land ready (dispatch order)
    assert [t["id"] for t in ready] == [b["id"], a["id"]]
