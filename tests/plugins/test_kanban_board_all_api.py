"""Cross-board query: GET /board/all (issue #121549 "All boards" view).

Read-only first step of the dashboard's "All boards" option: the same
columns from every non-archived board, each card tagged with its board
slug so the dashboard can label cards and filter by board, and the
attention strip aggregates diagnostics across boards.
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
    spec = importlib.util.spec_from_file_location(
        "hermes_kanban_board_all_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
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


def _cards(data: dict) -> list[dict]:
    return [t for col in data["columns"] for t in col["tasks"]]


def test_board_all_returns_cards_from_multiple_boards_with_board_slug(client):
    """One query, one payload: cards from every board, each tagged `board`."""
    client.post("/api/plugins/kanban/tasks", json={"title": "default board card"})
    kb.create_board("alpha")
    client.post("/api/plugins/kanban/tasks?board=alpha", json={"title": "alpha card"})
    kb.create_board("beta")
    client.post("/api/plugins/kanban/tasks?board=beta", json={"title": "beta card"})

    r = client.get("/api/plugins/kanban/board/all")
    assert r.status_code == 200, r.text
    data = r.json()

    # Same column layout as /board so the dashboard reuses its renderer.
    names = [c["name"] for c in data["columns"]]
    assert set(names) >= {"triage", "todo", "ready", "running", "blocked", "done"}

    by_title = {t["title"]: t for t in _cards(data)}
    assert {"default board card", "alpha card", "beta card"} <= set(by_title), by_title
    # Every card carries its board slug — the tag the dashboard renders.
    assert all("board" in t for t in _cards(data)), _cards(data)
    assert by_title["default board card"]["board"] == "default"
    assert by_title["alpha card"]["board"] == "alpha"
    assert by_title["beta card"]["board"] == "beta"

    # Per-board rollups are merged across boards (the banner/filter read them).
    assert "alpha" in data["tenants"] or data["tenants"] == []
    assert isinstance(data["latest_event_id"], int)


def test_board_all_excludes_archived_boards(client):
    """Archived boards are invisible to the cross-board query, whichever way
    they were archived: metadata flag (dir still on disk) or dir moved to
    ``boards/_archived/``."""
    kb.create_board("live")
    client.post("/api/plugins/kanban/tasks?board=live", json={"title": "live card"})

    # Flag-archived: directory stays put, board.json says archived.
    kb.create_board("retired")
    client.post("/api/plugins/kanban/tasks?board=retired", json={"title": "retired card"})
    kb.write_board_metadata("retired", archived=True)

    # Moved-archived: remove_board(archive=True) renames the dir into _archived/.
    kb.create_board("gone")
    client.post("/api/plugins/kanban/tasks?board=gone", json={"title": "gone card"})
    kb.remove_board("gone")

    r = client.get("/api/plugins/kanban/board/all")
    assert r.status_code == 200, r.text
    data = r.json()

    titles = {t["title"] for t in _cards(data)}
    assert "live card" in titles
    assert "retired card" not in titles
    assert "gone card" not in titles
    boards = {t.get("board") for t in _cards(data)}
    assert "live" in boards
    assert not boards & {"retired", "gone"}
