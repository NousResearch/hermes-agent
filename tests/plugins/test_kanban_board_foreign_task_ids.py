"""Kanban dashboard plugin: task ids leave the API as strings, whatever the board's ``tasks.id`` column holds.

Hermes writes ``t_<hex>`` text ids, but a board seeded outside Hermes (#112953) may declare ``tasks.id`` INTEGER or
leave it untyped, and SQLite then hands back numbers. The desktop board calls string methods on every card's id
(``shortId``), so one numeric id took the whole page down with ``(e ?? "").replace is not a function``.
"""

from __future__ import annotations

import importlib.util
import sqlite3
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
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_foreign_ids_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


def _seeded_client(tmp_path, monkeypatch, id_column: str) -> tuple[TestClient, Path]:
    """A default board whose ``tasks`` table an external harness created with ``id_column``, holding task 1 (todo)
    and task 2 (done)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = kb.kanban_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    seed = sqlite3.connect(path)
    seed.execute(f"CREATE TABLE tasks ({id_column}, title TEXT NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL)")
    seed.executemany("INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, ?, ?)",
                     [(1, "seeded todo", "todo", now), (2, "seeded done", "done", now)])
    seed.commit()
    seed.close()
    kb.init_db()  # Hermes heals the reduced schema; the id column keeps its foreign type
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE tasks SET completed_at = ? WHERE id = 2", (now,))
        assert conn.execute("SELECT typeof(id) FROM tasks WHERE id = 1").fetchone() == ("integer",)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app), path


def _column_ids(board: dict, name: str) -> list:
    return [t["id"] for t in next(c["tasks"] for c in board["columns"] if c["name"] == name)]


def test_board_sends_string_ids_for_a_board_seeded_with_numeric_ids(tmp_path, monkeypatch):
    client, _ = _seeded_client(tmp_path, monkeypatch, "id PRIMARY KEY")  # untyped: Hermes can still add t_ tasks
    made = client.post("/api/plugins/kanban/tasks", json={"title": "made by hermes"}).json()["task"]["id"]
    finished = client.patch(f"/api/plugins/kanban/tasks/{made}",
                            json={"status": "done", "result": "done", "summary": "done"})
    assert finished.status_code == 200, finished.text

    r = client.get("/api/plugins/kanban/board")

    assert r.status_code == 200, r.text  # sorting a done column that mixes int and str ids raised TypeError
    every = [i for c in r.json()["columns"] for i in _column_ids(r.json(), c["name"])]
    assert made.startswith("t_") and sorted(every) == sorted(["1", "2", made])
    assert sorted(_column_ids(r.json(), "done")) == sorted(["2", made])


def test_task_detail_sends_string_ids_for_the_task_and_its_linked_tasks(tmp_path, monkeypatch):
    client, path = _seeded_client(tmp_path, monkeypatch, "id INTEGER PRIMARY KEY")
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (1, 2)")

    r = client.get("/api/plugins/kanban/tasks/1")

    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["task"]["id"] == "1"
    assert [row["id"] for row in detail["link_tasks"]] == ["2"]
    assert [row["id"] for row in detail["child_results"]] == ["2"]
