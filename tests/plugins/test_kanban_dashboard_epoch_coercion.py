"""Kanban dashboard plugin: INTEGER timestamp columns carrying legacy ISO text.

Regression for the dashboard 500 "Failed to load Kanban board: Internal Server
Error": a migrated/legacy board can hold ISO-8601 TEXT in columns the schema
declares INTEGER (pre-epoch writers). The done-column sort then ran
``-(d["completed_at"] or 0)`` on a str → TypeError → FastAPI 500. The plugin
now coerces timestamp fields through ``kanban_db._to_epoch`` at serialization,
so the board loads and cards expose epoch ints.
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

_LEGACY_TEXT_TS = "2026-07-19T18:41:15.945797+00:00"  # epoch 1784486475


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_epoch_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def plugin_mod(kanban_home):
    return _load_plugin_router()


@pytest.fixture
def client(plugin_mod):
    app = FastAPI()
    app.include_router(plugin_mod.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _make_done_task_with_text_completed_at():
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, created_by="test", title="legacy text timestamp")
        # Simulate a pre-epoch writer storing ISO text in an INTEGER column.
        conn.execute(
            "UPDATE tasks SET status='done', completed_at=? WHERE id=?",
            (_LEGACY_TEXT_TS, task_id),
        )
        conn.commit()
        return task_id


def test_board_loads_with_text_completed_at(client):
    _make_done_task_with_text_completed_at()

    resp = client.get("/api/plugins/kanban/board")
    assert resp.status_code == 200, resp.text

    done_column = next(c for c in resp.json()["columns"] if c["name"] == "done")
    card = next(t for t in done_column["tasks"] if t["title"] == "legacy text timestamp")
    # Coerced to an epoch int, not the raw ISO string (would crash the sort).
    assert card["completed_at"] == 1784486475
    assert isinstance(card["completed_at"], int)


def test_task_dict_coerces_text_timestamps(plugin_mod):
    mod = plugin_mod
    task = kb.Task(
        id="t_text_ts",
        title="legacy",
        body=None,
        assignee=None,
        status="done",
        priority=0,
        created_by="test",
        created_at="2026-07-19T18:41:15.945797+00:00",  # type: ignore[arg-type]  # legacy drift under test
        started_at=None,
        completed_at=_LEGACY_TEXT_TS,  # type: ignore[arg-type]  # legacy drift under test
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    d = mod._task_dict(task)
    assert d["created_at"] == 1784486475
    assert d["completed_at"] == 1784486475
    assert d["started_at"] is None

    # The exact done-column sort expression must now be safe on the dict.
    sorted([d], key=lambda x: (x["completed_at"] is None, -(x["completed_at"] or 0)))
