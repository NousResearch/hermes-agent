"""Kanban dashboard plugin: the done column can be capped.

The done column is completion history and grows without bound (one busy board
reached 12,000 done cards: a ~49 MB ``GET /board`` that every desktop client
re-fetched for its status-bar count). ``done_limit`` (or the
``dashboard.kanban.board_done_limit`` config default) returns only the newest
done cards, ``done_total`` keeps the full count, and per-card rollups are
computed only for the cards actually returned.
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


def _load_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_done_limit_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return _load_plugin()


@pytest.fixture
def client(plugin):
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _create(client, title):
    r = client.post("/api/plugins/kanban/tasks", json={"title": title})
    assert r.status_code == 200, r.text
    return r.json()["task"]["id"]


def _finish(client, task_id, completed_at):
    r = client.patch(f"/api/plugins/kanban/tasks/{task_id}",
                     json={"status": "done", "result": "done", "summary": "done"})
    assert r.status_code == 200, r.text
    with kbc.connect() as conn:
        conn.execute("UPDATE tasks SET completed_at = ? WHERE id = ?", (completed_at, task_id))
        conn.commit()


@pytest.fixture
def seeded(client):
    """Five done cards (completed 1000..5000) and one open card."""
    done = [_create(client, f"done {i}") for i in range(5)]
    for i, task_id in enumerate(done):
        _finish(client, task_id, 1000 * (i + 1))
    open_id = _create(client, "still open")
    newest_first = list(reversed(done))
    return newest_first, open_id


def _board(client, **params):
    r = client.get("/api/plugins/kanban/board", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _ids(board, column):
    return [t["id"] for c in board["columns"] if c["name"] == column for t in c["tasks"]]


def test_default_returns_every_done_card_with_total(client, seeded):
    newest_first, open_id = seeded
    board = _board(client)
    assert _ids(board, "done") == newest_first
    assert board["done_total"] == 5
    assert board["done_limit"] is None
    assert open_id in [t for c in board["columns"] if c["name"] != "done" for t in _ids(board, c["name"])]


def test_done_limit_keeps_newest_completed_and_open_lanes(client, seeded):
    newest_first, open_id = seeded
    board = _board(client, done_limit=2)
    assert _ids(board, "done") == newest_first[:2]
    assert board["done_total"] == 5
    assert board["done_limit"] == 2
    open_lanes = [t for c in board["columns"] if c["name"] != "done" for t in _ids(board, c["name"])]
    assert open_lanes == [open_id]


def test_done_limit_zero_means_all(client, seeded):
    newest_first, _ = seeded
    board = _board(client, done_limit=0)
    assert _ids(board, "done") == newest_first
    assert board["done_limit"] is None


def test_negative_done_limit_rejected(client, seeded):
    assert client.get("/api/plugins/kanban/board", params={"done_limit": -1}).status_code == 422


def test_config_default_applies_and_request_overrides(plugin, client, seeded, monkeypatch):
    newest_first, _ = seeded
    monkeypatch.setattr(plugin, "_load_config_or_empty",
                        lambda: {"dashboard": {"kanban": {"board_done_limit": 3}}})
    board = _board(client)
    assert _ids(board, "done") == newest_first[:3]
    assert (board["done_total"], board["done_limit"]) == (5, 3)
    assert _ids(_board(client, done_limit=1), "done") == newest_first[:1]
    assert len(_ids(_board(client, done_limit=0), "done")) == 5


def test_rollups_cover_only_returned_cards(plugin, client, seeded, monkeypatch):
    newest_first, open_id = seeded
    seen: list = []
    real = plugin._compute_task_diagnostics

    def spy(conn, task_ids=None):
        seen.append(None if task_ids is None else sorted(task_ids))
        return real(conn, task_ids=task_ids)

    monkeypatch.setattr(plugin, "_compute_task_diagnostics", spy)
    _board(client, done_limit=2)
    assert seen[-1] == sorted([open_id, *newest_first[:2]])
    # Nothing dropped: the whole-board query, exactly as before.
    _board(client)
    assert seen[-1] is None
