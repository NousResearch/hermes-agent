"""Archiving a *cancelled* plan must not release its unfinished children.

``archived`` is terminal for dependency purposes, so ``recompute_ready`` treats it
exactly like ``done``. That is correct when a finished card is later archived, and
wrong when a plan is **cancelled**: the child was never approved to run on its own,
yet archiving the parent promotes it to ``ready`` and a worker picks it up.

These tests pin the distinction:

* archiving a parent that is not ``done`` parks its unfinished children (sticky
  ``blocked``), so ``recompute_ready`` cannot promote them;
* archiving a parent that *is* ``done`` keeps the old promotion behaviour;
* ``cascade=True`` archives the whole subtree instead, leaves first, and leaves
  nothing behind to hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_cancelled_parent_holds_unfinished_child(kanban_home: Path) -> None:
    """A child of a cancelled (archived while not done) parent must not be promoted."""
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="plan: publish session")
        child = kb.create_task(conn, title="do the publish", parents=[parent])
        assert kb.get_task(conn, child).status != "archived"

        assert kb.archive_task(conn, parent) is True

        assert kb.get_task(conn, child).status == "blocked", "unfinished child was released"
        # Sticky: the dispatcher's promotion pass must leave it alone.
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, child).status == "blocked"


def test_child_records_the_cancellation_reason(kanban_home: Path) -> None:
    """The held child carries an event naming the parent, so a human can act on it."""
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="plan: publish session")
        child = kb.create_task(conn, title="do the publish", parents=[parent])
        kb.archive_task(conn, parent)

        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (child,),
        ).fetchall()
        assert rows, "no blocked event recorded for the held child"
        payload = rows[-1]["payload"]
        assert parent in payload, f"reason must name the cancelled parent: {payload!r}"


def test_done_parent_archived_still_promotes_child(kanban_home: Path) -> None:
    """Archiving a *finished* parent keeps the documented promotion behaviour."""
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="finished step")
        child = kb.create_task(conn, title="next step", parents=[parent])

        # Drive the parent to done through the normal lifecycle.
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="step finished") is True
        assert kb.archive_task(conn, parent) is True

        assert kb.get_task(conn, child).status == "ready", "a done parent must still release its child"


def test_cascade_archives_the_whole_subtree(kanban_home: Path) -> None:
    """``cascade=True`` is the explicit "this whole branch is dead" choice."""
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="plan root")
        child = kb.create_task(conn, title="step two", parents=[parent])
        grandchild = kb.create_task(conn, title="step three", parents=[child])

        assert kb.archive_task(conn, parent, cascade=True) is True

        for tid in (parent, child, grandchild):
            assert kb.get_task(conn, tid).status == "archived", f"{tid} not archived by cascade"
