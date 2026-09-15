"""Kanban dashboard plugin: board-header dispatcher toggles.

PATCH /boards/{slug} with dispatch_enabled / auto_decompose_enabled /
review_dispatch_enabled — tri-state (``None`` = leave unchanged), same
convention as default_workdir/project_id. Attaches the plugin router to a
bare FastAPI app, as in test_kanban_dashboard_plugin.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_header_toggles_test", plugin_file)
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


def test_new_board_defaults_omit_toggle_overrides(client):
    r = client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    assert r.status_code == 200, r.text
    board = r.json()["board"]
    # No override written yet — the tri-state keys are absent, letting the
    # frontend/dispatcher fall back to their own defaults.
    assert "dispatch_enabled" not in board
    assert "auto_decompose_enabled" not in board
    assert "review_dispatch_enabled" not in board


def test_patch_sets_one_flag_without_touching_others(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})

    r = client.patch("/api/plugins/kanban/boards/widget", json={"dispatch_enabled": False})
    assert r.status_code == 200, r.text
    board = r.json()["board"]
    assert board["dispatch_enabled"] is False
    assert "auto_decompose_enabled" not in board
    assert "review_dispatch_enabled" not in board


def test_unrelated_patch_preserves_existing_overrides(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    client.patch(
        "/api/plugins/kanban/boards/widget",
        json={"dispatch_enabled": False, "auto_decompose_enabled": True, "review_dispatch_enabled": False},
    )

    # An unrelated field write (rename) must not disturb the toggles set above.
    r = client.patch("/api/plugins/kanban/boards/widget", json={"name": "Widget Renamed"})
    assert r.status_code == 200, r.text
    board = r.json()["board"]
    assert board["name"] == "Widget Renamed"
    assert board["dispatch_enabled"] is False
    assert board["auto_decompose_enabled"] is True
    assert board["review_dispatch_enabled"] is False


def test_patch_can_flip_each_flag_independently(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})

    r = client.patch("/api/plugins/kanban/boards/widget", json={"auto_decompose_enabled": False})
    assert r.status_code == 200
    board = r.json()["board"]
    assert board["auto_decompose_enabled"] is False
    assert "dispatch_enabled" not in board
    assert "review_dispatch_enabled" not in board

    r = client.patch("/api/plugins/kanban/boards/widget", json={"review_dispatch_enabled": False})
    assert r.status_code == 200
    board = r.json()["board"]
    assert board["review_dispatch_enabled"] is False
    # Prior write is preserved across this second independent write.
    assert board["auto_decompose_enabled"] is False
    assert "dispatch_enabled" not in board


def test_boards_list_surfaces_toggle_overrides(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    client.patch("/api/plugins/kanban/boards/widget", json={"dispatch_enabled": False})

    widget = next(b for b in client.get("/api/plugins/kanban/boards").json()["boards"] if b["slug"] == "widget")
    assert widget["dispatch_enabled"] is False
