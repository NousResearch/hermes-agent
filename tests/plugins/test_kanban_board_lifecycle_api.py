"""Kanban dashboard plugin: board rename + delete over the REST surface.

The desktop board switcher drives both from its kebab menu and reads
``result.new_path`` back to tell the user where an archived board went, so
these assert the response shape as well as the on-disk outcome.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_lifecycle_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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


def test_rename_board_keeps_slug(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})

    r = client.patch("/api/plugins/kanban/boards/widget", json={"name": "Gadget"})
    assert r.status_code == 200, r.text
    assert r.json()["board"] == {**r.json()["board"], "slug": "widget", "name": "Gadget"}

    listed = next(b for b in client.get("/api/plugins/kanban/boards").json()["boards"] if b["slug"] == "widget")
    assert listed["name"] == "Gadget"


def test_delete_board_archives_and_reverts_current(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    client.post("/api/plugins/kanban/tasks?board=widget", json={"title": "do the thing"})
    kb.set_current_board("widget")

    r = client.delete("/api/plugins/kanban/boards/widget")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"]["action"] == "archived"
    assert body["current"] == "default"

    # The switcher shows this path, so it must point at real recoverable data.
    archived = Path(body["result"]["new_path"])
    assert archived.is_dir()
    assert (archived / "kanban.db").exists()
    # The original slug keeps only an archived tombstone (no DB) — a stale
    # dashboard tab must not resurrect an empty board there (#43243).
    assert not (kb.board_dir("widget") / "kanban.db").exists()
    assert kb.read_board_metadata("widget")["archived"] is True

    assert all(b["slug"] != "widget" for b in client.get("/api/plugins/kanban/boards").json()["boards"])


def test_delete_board_hard_leaves_nothing(client):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget"})

    r = client.delete("/api/plugins/kanban/boards/widget?delete=true")
    assert r.status_code == 200, r.text
    assert r.json()["result"]["action"] == "deleted"
    assert not kb.board_dir("widget").exists()


def test_delete_default_board_is_refused(client):
    r = client.delete("/api/plugins/kanban/boards/default")
    assert r.status_code == 400
    assert "default" in r.json()["detail"]


@pytest.mark.parametrize("hard_delete", [False, True], ids=["archive", "hard-delete"])
def test_delete_board_locked_dir_is_a_400_not_a_500(client, monkeypatch, hard_delete):
    """#143908: a board dir held open (the gateway's kanban.db handle) must surface
    as an actionable 400, not a raw 500 — and must not half-remove the board."""
    from hermes_cli import kanban_db_boards as kbb

    client.post("/api/plugins/kanban/boards", json={"slug": "locked", "name": "Locked"})
    monkeypatch.setattr(kbb, "_DIR_OP_RETRY_DELAY_S", 0.0, raising=False)  # no wall-clock waits

    real_rename, real_rmtree = Path.rename, shutil.rmtree
    state = {"locked": True}

    def flaky_rename(self, target):
        if state["locked"]:
            raise PermissionError(13, "The process cannot access the file", str(self))
        return real_rename(self, target)

    def flaky_rmtree(path, *args, **kwargs):
        if state["locked"]:
            raise PermissionError(13, "The process cannot access the file", str(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", flaky_rename)  # archive branch
    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)  # hard-delete branch

    url = "/api/plugins/kanban/boards/locked" + ("?delete=true" if hard_delete else "")
    r = client.delete(url)

    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert f"could not {'delete' if hard_delete else 'archive'} board 'locked'" in detail
    assert "hermes gateway restart" in detail
    # Still there, still a live board: the failed removal changed nothing.
    assert kb.board_dir("locked").is_dir()
    assert kb.read_board_metadata("locked")["archived"] is False
    assert "locked" in [b["slug"] for b in kb.list_boards(include_archived=False)]

    # Lock released (what `hermes gateway restart` does): the same request works.
    state["locked"] = False
    r = client.delete(url)
    assert r.status_code == 200, r.text
