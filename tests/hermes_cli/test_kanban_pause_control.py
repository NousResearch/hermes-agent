"""Durable, fail-closed state for the Kanban-only dispatch control."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_control
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    connection = kbc.connect()
    try:
        yield connection
    finally:
        with contextlib.suppress(sqlite3.ProgrammingError):
            connection.close()


def test_missing_control_row_is_running_revision_zero(conn):
    before = conn.total_changes

    state = kanban_control.read_control(conn)

    assert state.state == "running"
    assert state.revision == 0
    assert state.changed_at is None
    assert state.changed_by is None
    assert state.reason is None
    assert state.error_code is None
    assert conn.total_changes == before


def test_pause_persists_and_records_one_matching_event(conn):
    paused = kanban_control.pause_board(
        conn, changed_by="operator@example", reason="maintenance",
    )
    conn.close()

    reopened = kbc.connect()
    try:
        observed = kanban_control.read_control(reopened)
        events = reopened.execute(
            "SELECT revision, old_state, new_state, changed_by, reason, changed_at "
            "FROM kanban_dispatch_control_events ORDER BY revision"
        ).fetchall()
    finally:
        reopened.close()

    assert paused == observed
    assert observed.state == "paused"
    assert observed.revision == 1
    assert observed.changed_by == "operator@example"
    assert observed.reason == "maintenance"
    assert len(events) == 1
    assert dict(events[0]) == {
        "revision": observed.revision,
        "old_state": "running",
        "new_state": "paused",
        "changed_by": observed.changed_by,
        "reason": observed.reason,
        "changed_at": observed.changed_at,
    }


def test_repeated_pause_is_idempotent_and_resume_advances_revision(conn):
    first = kanban_control.pause_board(conn, changed_by="one", reason="hold")
    repeated = kanban_control.pause_board(conn, changed_by="two", reason="changed text")
    resumed = kanban_control.resume_board(conn, changed_by="three", reason="clear")
    conn.close()

    reopened = kbc.connect()
    try:
        observed = kanban_control.read_control(reopened)
        events = reopened.execute(
            "SELECT revision, old_state, new_state FROM kanban_dispatch_control_events "
            "ORDER BY revision"
        ).fetchall()
    finally:
        reopened.close()

    assert repeated == first
    assert resumed == observed
    assert observed.state == "running"
    assert observed.revision == 2
    assert [(row["revision"], row["old_state"], row["new_state"]) for row in events] == [
        (1, "running", "paused"),
        (2, "paused", "running"),
    ]


@pytest.mark.parametrize(
    "corrupt_sql",
    [
        "UPDATE kanban_dispatch_control SET state='stopped' WHERE id=1",
        "UPDATE kanban_dispatch_control SET revision=-1 WHERE id=1",
    ],
)
def test_invalid_control_row_reads_unknown_without_repair(conn, corrupt_sql):
    kanban_control.pause_board(conn, changed_by="operator")
    conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.execute(corrupt_sql)
    conn.execute("PRAGMA ignore_check_constraints=OFF")
    before = conn.total_changes

    state = kanban_control.read_control(conn)

    assert state.state == "unknown"
    assert state.error_code == "invalid_control_state"
    assert conn.total_changes == before


def test_schema_drift_reads_unknown_without_repair(conn):
    conn.execute("DROP TABLE kanban_dispatch_control")
    conn.execute("CREATE TABLE kanban_dispatch_control (id INTEGER PRIMARY KEY, state TEXT)")
    conn.execute("INSERT INTO kanban_dispatch_control VALUES (1, 'paused')")
    before = conn.total_changes

    state = kanban_control.read_control(conn)

    assert state.state == "unknown"
    assert state.error_code == "control_schema_invalid"
    assert conn.total_changes == before
