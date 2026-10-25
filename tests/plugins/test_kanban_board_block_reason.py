"""Kanban dashboard plugin: ``GET /board`` says why a card waits, when each card
last changed, and which cards it links to.

The block reason lives only in the payload of the task's newest block event,
the per-task newest event id only in ``task_events``, and the link ids only in
``task_links`` (which the board already reads for ``link_counts``). These pin
all three onto each card so one board read is enough for a "waiting on you"
view, without opening ``kanban.db`` or calling ``/tasks/{id}`` per card.
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


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_block_reason_test", plugin_file)
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


def _cards(client):
    board = client.get("/api/plugins/kanban/board").json()
    return {t["id"]: t for column in board["columns"] for t in column["tasks"]}


def _create(title, **kwargs):
    with kbc.connect() as conn:
        return kb.create_task(conn, title=title, **kwargs)


def test_blocked_card_carries_its_reason_and_block_event(client):
    tid = _create("needs a decision")
    with kbc.connect() as conn:
        assert kb.block_task(conn, tid, reason="Pick A or B", kind="needs_input")

    card = _cards(client)[tid]
    assert card["status"] == "blocked"
    assert card["block_reason"] == "Pick A or B"
    assert card["block_event"]["kind"] == "blocked"
    assert isinstance(card["block_event"]["id"], int) and card["block_event"]["at"] > 0


def test_repeated_block_lands_in_triage_with_the_newest_reason(client):
    tid = _create("asks twice")
    with kbc.connect() as conn:
        assert kb.block_task(conn, tid, reason="first ask", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        assert kb.block_task(conn, tid, reason="second ask", kind="needs_input")

    card = _cards(client)[tid]
    assert card["status"] == "triage"
    assert card["block_event"]["kind"] == "block_loop_detected"
    assert card["block_reason"] == "second ask"


def test_card_created_blocked_names_its_creator(client):
    tid = _create("parked on create", initial_status="blocked", created_by="gemma")

    card = _cards(client)[tid]
    assert card["block_reason"] == "initial_status"
    assert card["block_event"]["actor"] == "gemma"


def test_card_not_waiting_has_no_block_reason_even_if_it_was_blocked_before(client):
    tid = _create("blocked once")
    with kbc.connect() as conn:
        assert kb.block_task(conn, tid, reason="old question", kind="needs_input")
        assert kb.unblock_task(conn, tid)

    card = _cards(client)[tid]
    assert card["status"] != "blocked"
    assert card["block_reason"] is None
    assert card["block_event"] is None


def test_links_list_the_ids_link_counts_counts(client):
    parent = _create("parent")
    child = _create("child")
    with kbc.connect() as conn:
        assert kb.link_tasks(conn, parent, child)

    cards = _cards(client)
    assert cards[parent]["links"] == {"parents": [], "children": [child]}
    assert cards[child]["links"] == {"parents": [parent], "children": []}
    assert cards[parent]["link_counts"] == {"parents": 0, "children": 1}


def test_each_card_has_its_own_latest_event_id_that_moves_only_with_its_events(client):
    a = _create("a")
    b = _create("b")
    before = _cards(client)
    with kbc.connect() as conn:
        newest = {r["task_id"]: r["m"] for r in conn.execute(
            "SELECT task_id, MAX(id) AS m FROM task_events GROUP BY task_id")}
        assert kb.block_task(conn, a, reason="why", kind="needs_input")
    after = _cards(client)

    assert before[a]["latest_event_id"] == newest.get(a, 0)
    assert after[a]["latest_event_id"] > before[a]["latest_event_id"]
    assert after[b]["latest_event_id"] == before[b]["latest_event_id"]
