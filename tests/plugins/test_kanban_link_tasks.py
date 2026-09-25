"""GET /tasks/:id includes `link_tasks` — {id, title, status} rows for every
parent/child — so the desktop drawer renders titles instead of raw ids
(#kanban-modal). Same router-on-bare-FastAPI harness as the attachments tests.
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
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_linktasks_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._dashboard_router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def api(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_link_tasks_resolves_titles_both_sides(kanban_home, api):
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="Parent title")
        child = kb.create_task(conn, title="Child title")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
    finally:
        conn.close()

    # The child's detail: parents resolve, children stay empty.
    detail = api.get(f"/api/plugins/kanban/tasks/{child}").json()
    assert detail["links"] == {"parents": [parent], "children": []}
    assert [(row["id"], row["title"]) for row in detail["link_tasks"]] == [(parent, "Parent title")]
    assert detail["link_tasks"][0]["status"]

    # The parent's detail: children resolve.
    detail = api.get(f"/api/plugins/kanban/tasks/{parent}").json()
    assert detail["links"] == {"parents": [], "children": [child]}
    assert [(row["id"], row["title"]) for row in detail["link_tasks"]] == [(child, "Child title")]


def test_link_tasks_empty_when_no_links(kanban_home, api):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="Loner")
    finally:
        conn.close()

    detail = api.get(f"/api/plugins/kanban/tasks/{task_id}").json()
    assert detail["links"] == {"parents": [], "children": []}
    assert detail["link_tasks"] == []
