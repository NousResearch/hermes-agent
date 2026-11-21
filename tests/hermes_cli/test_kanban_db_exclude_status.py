"""Negative status filter (``exclude_status``) for the Kanban task listing.

Lives in its own module because ``tests/hermes_cli/test_kanban_db.py`` is past
the repo's file-line cap and may only shrink — new tests go into a topical
sibling (see root CONTRIBUTING.md, "Code health ratchet").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_exclude_status_drops_finished_work(kanban_home):
    """A board review needs "everything still in flight", which the single-value
    ``status`` filter cannot express: it can only include, never exclude."""
    conn = kbc.connect()
    try:
        live = kb.create_task(conn, title="still live")
        finished = kb.create_task(conn, title="finished")
        shelved = kb.create_task(conn, title="shelved")
        kb.complete_task(conn, finished, summary="done")
        kb.archive_task(conn, shelved)
        ids = {t.id for t in kb.list_tasks(conn, exclude_status=["done", "archived"])}
        assert ids == {live}
        # Sanity: without the filter the same call still carries the finished card.
        assert {t.id for t in kb.list_tasks(conn)} == {live, finished}
    finally:
        conn.close()


def test_exclude_status_can_still_include_archived(kanban_home):
    conn = kbc.connect()
    try:
        live = kb.create_task(conn, title="live")
        shelved = kb.create_task(conn, title="shelved")
        kb.archive_task(conn, shelved)
        ids = {
            t.id
            for t in kb.list_tasks(conn, include_archived=True, exclude_status=["done"])
        }
        assert ids == {live, shelved}
    finally:
        conn.close()


def test_exclude_status_rejects_unknown_status(kanban_home):
    """A typo must be an error, not a filter that silently matches everything."""
    conn = kbc.connect()
    try:
        kb.create_task(conn, title="anything")
        with pytest.raises(ValueError) as exc:
            kb.list_tasks(conn, exclude_status=["done", "nonsense"])
        assert "exclude_status" in str(exc.value)
    finally:
        conn.close()


def test_status_and_exclude_status_combine_as_and(kanban_home):
    """Documented resolution: a status named in BOTH filters is excluded."""
    conn = kbc.connect()
    try:
        kb.create_task(conn, title="ready one")
        assert kb.list_tasks(conn, status="ready", exclude_status=["ready"]) == []
    finally:
        conn.close()
