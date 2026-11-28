"""A task an older version left as ``cancelled`` or ``failed`` is finished work, so the dashboard shows it with the
archive and never as a phantom ``todo`` card, agreeing with ``hermes kanban stats``."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tests.plugins.test_kanban_dashboard_plugin import _load_plugin_router


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        ids = {status: kb.create_task(conn, title=status) for status in ("cancelled", "failed", "mystery")}
        ids["todo"] = kb.create_task(conn, title="still to do", parents=[ids["mystery"]])
        for status in ("cancelled", "failed", "mystery"):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, ids[status]))
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    client = TestClient(app)

    def fetch(**params) -> dict[str, list[str]]:
        data = client.get("/api/plugins/kanban/board", params=params).json()
        return {col["name"]: [card["title"] for card in col["tasks"]] for col in data["columns"]}
    return fetch


def test_retired_statuses_are_not_todo_cards(board):
    columns = board()

    assert "cancelled" not in columns["todo"] and "failed" not in columns["todo"]
    assert not any("cancelled" in titles or "failed" in titles for titles in columns.values())


def test_retired_statuses_show_under_the_archive_toggle(board):
    columns = board(include_archived="true")

    assert sorted(columns["archived"]) == ["cancelled", "failed"]
    assert "cancelled" not in columns["todo"]


def test_a_status_with_no_known_meaning_stays_visible_in_todo(board):
    assert "mystery" in board()["todo"]
