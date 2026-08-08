"""Regression tests for the task_lifecycle cluster (s4-w1b extraction).

Covers archive / hard-delete moved verbatim from ``hermes_cli.kanban_db``
(cluster c1 / task_lifecycle) into ``hermes_cli.task_lifecycle``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli.task_lifecycle import (
    archive_task,
    delete_archived_task,
    delete_task,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Re-export parity
# ---------------------------------------------------------------------------


def test_moved_names_reexported_on_kanban_db_module():
    for name in ("archive_task", "delete_archived_task", "delete_task"):
        assert getattr(kb, name) is globals()[name], name


def test_direct_module_import_works():
    import hermes_cli.task_lifecycle as tl
    assert tl.archive_task is archive_task


# ---------------------------------------------------------------------------
# archive_task
# ---------------------------------------------------------------------------


def test_archive_task_archives_and_emits_event(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="to archive")
        assert archive_task(conn, t) is True
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["status"] == "archived"
        ev = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'archived'",
            (t,),
        ).fetchone()
        assert ev is not None


def test_archive_task_missing_returns_false(kanban_home):
    with kb.connect() as conn:
        assert archive_task(conn, "missing-task") is False


def test_archive_task_already_archived_returns_false(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="twice")
        assert archive_task(conn, t) is True
        assert archive_task(conn, t) is False


def test_archive_task_promotes_dependents(kanban_home):
    """Archiving a parent must unblock its children (recompute_ready)."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent, child),
        )
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (child,),
        )
        conn.commit()
        assert archive_task(conn, parent) is True
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (child,),
        ).fetchone()
        assert row["status"] == "ready"


# ---------------------------------------------------------------------------
# delete_archived_task
# ---------------------------------------------------------------------------


def test_delete_archived_task_removes_row(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="purge me")
        archive_task(conn, t)
        assert delete_archived_task(conn, t) is True
        assert kb.get_task(conn, t) is None


def test_delete_archived_task_refuses_non_archived(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="active")
        assert delete_archived_task(conn, t) is False
        assert kb.get_task(conn, t) is not None


def test_delete_archived_task_cascades_related_rows(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="cascade")
        conn.execute("INSERT INTO task_comments (task_id, author, body, created_at) "
                     "VALUES (?, 'a', 'b', 1)", (t,))
        conn.execute("INSERT INTO task_events (task_id, kind, created_at) "
                     "VALUES (?, 'created', 1)", (t,))
        conn.commit()
        archive_task(conn, t)
        assert delete_archived_task(conn, t) is True
        assert conn.execute(
            "SELECT 1 FROM task_comments WHERE task_id = ?", (t,),
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ?", (t,),
        ).fetchone() is None


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


def test_delete_task_hard_deletes(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="hard delete")
        assert delete_task(conn, t) is True
        assert kb.get_task(conn, t) is None


def test_delete_task_missing_returns_false(kanban_home):
    with kb.connect() as conn:
        assert delete_task(conn, "missing-task") is False


def test_delete_task_cascades_links_and_comments(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (a, b),
        )
        conn.execute("INSERT INTO task_comments (task_id, author, body, created_at) "
                     "VALUES (?, 'x', 'y', 1)", (a,))
        conn.commit()
        assert delete_task(conn, a) is True
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? OR child_id = ?",
            (a, a),
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM task_comments WHERE task_id = ?", (a,),
        ).fetchone() is None
