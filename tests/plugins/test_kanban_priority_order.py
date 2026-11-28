"""Default kanban ordering must honour the documented priority bands.

``kanban_create`` documents "LOWER = picked sooner (P0 > P1 > P2 > P3), FIFO
within band". Before the fix the default listing and the dispatch lane
queries used ``ORDER BY priority DESC``, so a P2 card outranked an earlier
P1 card and the dispatcher spawned lower-priority work first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return h


def _mk(conn, title: str, priority: int) -> str:
    return kb.create_task(conn, title=title, priority=priority, assignee="worker")


def test_default_list_orders_lower_priority_number_first(home):
    with kbc.connect() as conn:
        p2 = _mk(conn, "P2 created first", 2)
        p1 = _mk(conn, "P1 created second", 1)
        ids = [r.id for r in kb.list_tasks(conn)]
        assert ids.index(p1) < ids.index(p2), "P1 must outrank P2"


def test_fifo_within_the_same_priority_band(home):
    with kbc.connect() as conn:
        first = _mk(conn, "P1 first", 1)
        second = _mk(conn, "P1 second", 1)
        ids = [r.id for r in kb.list_tasks(conn)]
        assert ids.index(first) < ids.index(second)


def test_sort_key_priority_is_ascending():
    assert kb.VALID_SORT_ORDERS["priority"].startswith("priority ASC")
    assert kb.VALID_SORT_ORDERS["priority-desc"].startswith("priority DESC")
