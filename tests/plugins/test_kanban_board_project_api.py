"""Kanban dashboard plugin: project listing + project-scoped boards.

Attaches the plugin router to a bare FastAPI app (as in
test_kanban_dashboard_plugin.py) and exercises the project surface:
GET /projects, board create/patch/list carrying project scope, and a task
on a scoped board inheriting the project.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_proj_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / "widget-repo"
    repo.mkdir()
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="Widget", primary_path=str(repo))
    return {"id": pid, "primary_path": str(repo)}


def test_list_projects(client, project):
    r = client.get("/api/plugins/kanban/projects")
    assert r.status_code == 200
    hit = next(p for p in r.json()["projects"] if p["id"] == project["id"])
    assert hit["name"] == "Widget"
    assert hit["primary_path"] == project["primary_path"]


def test_create_board_with_project_mirrors_workdir(client, project):
    r = client.post(
        "/api/plugins/kanban/boards",
        json={"slug": "widget", "name": "Widget", "project_id": project["id"]},
    )
    assert r.status_code == 200, r.text
    board = r.json()["board"]
    assert board["project_id"] == project["id"]
    assert board["project_name"] == "Widget"
    assert board["default_workdir"] == project["primary_path"]


def test_create_board_rejects_unknown_project(client):
    r = client.post("/api/plugins/kanban/boards", json={"slug": "bad", "project_id": "p_nope"})
    assert r.status_code == 400


def test_patch_board_set_and_clear_project(client, project):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})

    r = client.patch("/api/plugins/kanban/boards/widget", json={"project_id": project["id"]})
    assert r.status_code == 200, r.text
    assert r.json()["board"]["project_id"] == project["id"]

    r = client.patch("/api/plugins/kanban/boards/widget", json={"project_id": ""})
    assert r.status_code == 200
    assert r.json()["board"]["project_id"] is None


def test_boards_list_surfaces_project(client, project):
    client.post(
        "/api/plugins/kanban/boards",
        json={"slug": "widget", "name": "Widget", "project_id": project["id"]},
    )
    widget = next(b for b in client.get("/api/plugins/kanban/boards").json()["boards"] if b["slug"] == "widget")
    assert widget["project_id"] == project["id"]
    assert widget["project_name"] == "Widget"


def test_task_on_scoped_board_inherits_project(client, project):
    client.post(
        "/api/plugins/kanban/boards",
        json={"slug": "widget", "name": "Widget", "project_id": project["id"]},
    )
    r = client.post("/api/plugins/kanban/tasks?board=widget", json={"title": "do the thing"})
    assert r.status_code == 200, r.text
    task_id = r.json()["task"]["id"]

    conn = kb.connect(board="widget")
    try:
        assert kb.get_task(conn, task_id).project_id == project["id"]
    finally:
        conn.close()
