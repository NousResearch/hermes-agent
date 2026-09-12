"""Regression test: `complete_task` must accept in_progress status.

Background: the kanban-done-proof-sweep reopens cards from `done` to
`in_progress` (`hermes_cli/kanban.py` _cmd_complete dispatcher). The
`complete_task` SQL UPDATE in `hermes_cli/kanban_db.py` filters on
`status IN ('running', 'ready', 'blocked', 'review')` — so cards in
`in_progress` (and any non-listed VALID_STATUS value) become unclosable.

This test pins the fix in place: in_progress MUST complete, terminal
states (done/archived) MUST NOT re-complete.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

# Use the LIVE kanban.db as a schema source (read-only). The test creates its
# own writable copy in tmp_path so tests don't mutate production state.
LIVE_KANBAN_DB = r"C:\Users\bbask\AppData\Local\hermes\kanban.db"


def _open_with_row_factory(db_path):
    """Open a connection with sqlite3.Row so dict-style index access works."""
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture
def kanban_db(tmp_path):
    """Copy live kanban.db to tmp_path so we test against the real schema
    without polluting production state. Wipe all data so tests start clean."""
    if not Path(LIVE_KANBAN_DB).exists():
        pytest.skip(f"Live kanban.db not present at {LIVE_KANBAN_DB}")
    db_path = tmp_path / "kanban.db"
    shutil.copy2(LIVE_KANBAN_DB, db_path)
    # Wipe all task data (preserve schema)
    conn = _open_with_row_factory(db_path)
    try:
        for table in ("task_attachments", "task_events", "task_comments",
                      "task_runs", "task_links", "tasks"):
            try:
                conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass  # table doesn't exist in this version
        conn.commit()
    finally:
        conn.close()

    sys.path.insert(0, r"C:\Users\bbask\AppData\Local\hermes\hermes-agent")
    from hermes_cli import kanban_db as kd
    return kd, db_path


def _insert_task(db_path, status, task_id="t_test"):
    """Insert a single task in the given status (no parents)."""
    conn = _open_with_row_factory(db_path)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, goal_mode) VALUES (?, 'test', ?, 1, 0)",
        (task_id, status),
    )
    conn.commit()
    conn.close()


def test_complete_task_accepts_in_progress(kanban_db):
    """Cards in `in_progress` (the proof-sweep reopen state) MUST complete."""
    kd, db_path = kanban_db
    _insert_task(db_path, "in_progress")

    conn = _open_with_row_factory(db_path)
    ok = kd.complete_task(conn, "t_test", result="verified", summary="audit")
    assert ok is True, "complete_task should accept in_progress status"

    row = conn.execute("SELECT status, result, completed_at FROM tasks WHERE id='t_test'").fetchone()
    assert row[0] == "done", f"status should be done, got {row[0]!r}"
    assert row[1] == "verified"
    assert row[2] is not None, "completed_at must be set"
    conn.close()


def test_complete_task_accepts_scheduled(kanban_db):
    """Cards in `scheduled` (waiting on time) MUST complete when manually forced."""
    kd, db_path = kanban_db
    _insert_task(db_path, "scheduled")

    conn = _open_with_row_factory(db_path)
    ok = kd.complete_task(conn, "t_test", result="forced complete")
    assert ok is True

    row = conn.execute("SELECT status FROM tasks WHERE id='t_test'").fetchone()
    assert row[0] == "done"
    conn.close()


def test_complete_task_accepts_triage(kanban_db):
    """Cards in `triage` (just-classified, awaiting action) MUST complete."""
    kd, db_path = kanban_db
    _insert_task(db_path, "triage")

    conn = _open_with_row_factory(db_path)
    ok = kd.complete_task(conn, "t_test", result="triaged")
    assert ok is True
    conn.close()


def test_complete_task_still_rejects_terminal_states(kanban_db):
    """Done and archived states are terminal — must NOT re-complete to done."""
    kd, db_path = kanban_db
    _insert_task(db_path, "done", task_id="t_terminal_done")
    _insert_task(db_path, "archived", task_id="t_terminal_archived")

    conn = _open_with_row_factory(db_path)
    ok_done = kd.complete_task(conn, "t_terminal_done", result="x")
    ok_arch = kd.complete_task(conn, "t_terminal_archived", result="x")
    assert ok_done is False, "should reject done (already terminal)"
    assert ok_arch is False, "should reject archived (already terminal)"
    conn.close()
