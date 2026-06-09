"""Workboard-facing Kanban dashboard API regression tests."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    mod_name = "hermes_dashboard_plugin_kanban_workboard_api_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name].router
    spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.router


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


def test_board_exposes_full_workboard_columns(client):
    resp = client.get("/api/plugins/kanban/board")
    assert resp.status_code == 200
    columns = [c["name"] for c in resp.json()["columns"]]
    assert columns == ["triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done"]


def test_board_cards_include_attachment_and_evidence_counts(client):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="ship preview")
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
        conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, "desktop.png", "/tmp/desktop.png", "image/png", 123, "test", int(time.time())),
        )
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (task_id, "test", "smoke passed", int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()

    resp = client.get("/api/plugins/kanban/board")
    assert resp.status_code == 200
    review = next(c for c in resp.json()["columns"] if c["name"] == "review")
    card = review["tasks"][0]
    assert card["attachment_count"] == 1
    assert card["comment_count"] == 1
    assert card["evidence_count"] == 2
