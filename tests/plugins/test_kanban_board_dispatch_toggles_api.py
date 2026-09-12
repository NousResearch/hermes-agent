"""Kanban dashboard plugin: per-board dispatch/auto-decompose/review-dispatch
overrides on the boards REST surface.

Attaches the plugin router to a bare FastAPI app (as in
test_kanban_dashboard_plugin.py) and exercises PATCH /boards/{slug} for the
three narrowing-only override fields added alongside the CLI's
`hermes kanban boards set-dispatch` / `set-auto-decompose` / `set-review-dispatch`.
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
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_dispatch_toggle_test", plugin_file)
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
def board(client):
    r = client.post("/api/plugins/kanban/boards", json={"slug": "proj"})
    assert r.status_code == 200
    return r.json()["board"]


def test_new_board_defaults_every_override_to_true(client, board):
    assert board["dispatch_enabled"] is True
    assert board["auto_decompose_enabled"] is True
    assert board["review_dispatch_enabled"] is True


def test_pre_existing_board_json_without_the_keys_defaults_to_true(client, kanban_home):
    # Simulate a board created before these fields existed: write a board.json
    # missing them entirely, then read it back through the same code path the
    # REST layer uses. Absent must mean "inherit the global switch" (true),
    # never a silent false.
    meta = kb.write_board_metadata("legacy")
    meta.pop("dispatch_enabled", None)
    meta.pop("auto_decompose_enabled", None)
    meta.pop("review_dispatch_enabled", None)
    import json
    Path(kb.board_metadata_path("legacy")).write_text(json.dumps(meta, indent=2))

    r = client.get("/api/plugins/kanban/boards")
    assert r.status_code == 200
    hit = next(b for b in r.json()["boards"] if b["slug"] == "legacy")
    assert hit["dispatch_enabled"] is True
    assert hit["auto_decompose_enabled"] is True
    assert hit["review_dispatch_enabled"] is True


def test_patch_turns_one_flag_off_and_leaves_the_others_untouched(client, board):
    r = client.patch("/api/plugins/kanban/boards/proj", json={"dispatch_enabled": False})
    assert r.status_code == 200
    updated = r.json()["board"]
    assert updated["dispatch_enabled"] is False
    assert updated["auto_decompose_enabled"] is True
    assert updated["review_dispatch_enabled"] is True

    # Read back through the standalone metadata API too (what the gateway
    # dispatcher actually calls at tick time), not just the REST response.
    meta = kb.read_board_metadata("proj")
    assert meta["dispatch_enabled"] is False
    assert meta["auto_decompose_enabled"] is True


def test_patch_omitting_the_fields_leaves_existing_overrides_alone(client, board):
    client.patch("/api/plugins/kanban/boards/proj", json={"dispatch_enabled": False})
    r = client.patch("/api/plugins/kanban/boards/proj", json={"name": "Renamed"})
    assert r.status_code == 200
    updated = r.json()["board"]
    assert updated["name"] == "Renamed"
    assert updated["dispatch_enabled"] is False  # untouched by the unrelated patch


def test_patch_can_turn_a_disabled_flag_back_on(client, board):
    client.patch("/api/plugins/kanban/boards/proj", json={"dispatch_enabled": False})
    r = client.patch("/api/plugins/kanban/boards/proj", json={"dispatch_enabled": True})
    assert r.status_code == 200
    assert r.json()["board"]["dispatch_enabled"] is True
