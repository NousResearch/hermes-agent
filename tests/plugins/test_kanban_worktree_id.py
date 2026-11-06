"""Dashboard REST surface for worktree_id (ORCA-06 kanban<->orca auto-link).

POST /tasks creates with a link, GET /tasks/:id (and /board) round-trips it,
PATCH /tasks/:id sets/replaces/clears it (``""`` or ``clear_worktree_id``),
invalid identifiers are a 400 with nothing persisted, and an omitted field
leaves an existing link untouched.
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


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_worktree_id_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _create(client, **kwargs):
    body = {"title": "task", "assignee": "worker"}
    body.update(kwargs)
    r = client.post("/api/plugins/kanban/tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()["task"]


def test_create_accepts_link(client):
    task = _create(client, worktree_id="wt_9f2c1a")
    assert task["worktree_id"] == "wt_9f2c1a"


def test_create_omitted_link_is_none(client):
    task = _create(client)
    assert task["worktree_id"] is None


def test_create_rejects_invalid_link(client):
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "t", "assignee": "w", "worktree_id": "has space"},
    )
    assert r.status_code == 400
    assert "worktree_id" in r.text


def test_get_roundtrips_link(client):
    task = _create(client, worktree_id="wt_9f2c1a")
    got = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]
    assert got["worktree_id"] == "wt_9f2c1a"


def test_board_response_carries_link(client):
    task = _create(client, worktree_id="wt_board")
    board = client.get("/api/plugins/kanban/board").json()
    flat = [card for col in board["columns"] for card in col["tasks"]]
    mine = next(c for c in flat if c["id"] == task["id"])
    assert mine["worktree_id"] == "wt_board"


def test_patch_sets_replaces_and_clears(client):
    task = _create(client)
    url = f"/api/plugins/kanban/tasks/{task['id']}"

    r = client.patch(url, json={"worktree_id": "wt_a"})
    assert r.status_code == 200, r.text
    assert r.json()["task"]["worktree_id"] == "wt_a"

    r = client.patch(url, json={"worktree_id": "wt_b"})
    assert r.json()["task"]["worktree_id"] == "wt_b"

    # Empty string is the explicit clear signal.
    r = client.patch(url, json={"worktree_id": ""})
    assert r.json()["task"]["worktree_id"] is None

    # The dedicated clear flag works without a value.
    client.patch(url, json={"worktree_id": "wt_c"})
    r = client.patch(url, json={"clear_worktree_id": True})
    assert r.json()["task"]["worktree_id"] is None


def test_patch_omitted_leaves_link_untouched(client):
    task = _create(client, worktree_id="wt_keep")
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}", json={"priority": 3})
    assert r.status_code == 200, r.text
    assert r.json()["task"]["worktree_id"] == "wt_keep"


def test_patch_rejects_invalid_link(client):
    task = _create(client)
    url = f"/api/plugins/kanban/tasks/{task['id']}"
    r = client.patch(url, json={"worktree_id": "no spaces"})
    assert r.status_code == 400
    got = client.get(url).json()["task"]
    assert got["worktree_id"] is None
