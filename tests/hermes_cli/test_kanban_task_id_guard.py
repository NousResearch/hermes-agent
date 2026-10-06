"""#119003: task ids are minted only by ``_new_task_id`` and never rewritten.
The schema enforces that for every writer, and init names triggers it didn't create."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# A trigger some tool left in a board: a blind status-summary rewrite. One
# routine claim with this in place reproduces #119003 exactly on a guard-less
# schema: the board collapses to ('t_running', '', 'running', 'mara') with no
# claim fields, every real task's history intact and integrity_check "ok".
ROGUE_TRIGGER = """
CREATE TRIGGER rogue AFTER UPDATE OF status ON tasks WHEN NEW.status = 'running' BEGIN
    DELETE FROM tasks;
    INSERT INTO tasks (id, title, status, assignee, created_at)
    VALUES ('t_' || NEW.status, '', NEW.status, 'mara', CAST(strftime('%s', 'now') AS INTEGER));
END;
"""


@pytest.fixture
def board(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return db_path


def _ids(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("SELECT id FROM tasks")}
    finally:
        conn.close()


def test_hermes_ids_pass_the_guard(board):
    with kbc.connect_closing() as conn:
        root = kb.create_task(conn, title="root", assignee="coder")
        child = kb.create_task(conn, title="child", assignee="coder", parents=[root])
    assert _ids(board) == {root, child}


@pytest.mark.parametrize("bad_id", ["t_running", "t_", "t_DEADBEEF", "task-1", "t_dead beef", None, 42])
def test_foreign_writer_cannot_insert_a_malformed_task_id(board, bad_id):
    with kbc.connect_closing() as conn:
        good = kb.create_task(conn, title="real", assignee="coder")
    # A plain connection, as a script or another process would open it.
    raw = sqlite3.connect(board)
    with pytest.raises(sqlite3.IntegrityError, match="malformed task id"):
        raw.execute(
            "INSERT INTO tasks (id, title, status, created_at) VALUES (?, '', 'running', 0)", (bad_id,)
        )
    raw.close()
    assert _ids(board) == {good}


def test_task_ids_are_immutable_but_rows_stay_editable(board):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="real", assignee="coder")
    raw = sqlite3.connect(board, isolation_level=None)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        raw.execute("UPDATE tasks SET id = 't_running' WHERE id = ?", (tid,))
    # Every other column (and a no-op id assignment) is untouched by the guard.
    raw.execute("UPDATE tasks SET id = id, title = 'edited', status = 'archived' WHERE id = ?", (tid,))
    raw.close()
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
    assert (task.title, task.status) == ("edited", "archived")


def test_foreign_trigger_repro_is_refused_and_rolled_back(board):
    with kbc.connect_closing() as conn:
        ids = {kb.create_task(conn, title=f"real {i}", assignee="coder") for i in range(5)}
        conn.executescript(ROGUE_TRIGGER)
        with pytest.raises(sqlite3.IntegrityError, match="see #119003"):
            kb.claim_task(conn, sorted(ids)[0])
    # The claim statement aborted as a whole: the trigger's DELETE went with it.
    assert _ids(board) == ids


def test_init_names_triggers_hermes_did_not_create(board, caplog):
    raw = sqlite3.connect(board)
    raw.executescript(ROGUE_TRIGGER)
    raw.close()
    kb._INITIALIZED_PATHS.discard(str(board.resolve()))
    caplog.set_level(logging.WARNING, logger=kb._log.name)

    with kbc.connect_closing():
        pass

    warnings = [r.getMessage() for r in caplog.records if "did not create" in r.getMessage()]
    assert len(warnings) == 1
    assert "'rogue'" in warnings[0] and "'tasks'" in warnings[0] and "#119003" in warnings[0]


def test_existing_board_gains_the_guards_on_connect(board):
    raw = sqlite3.connect(board)
    for name in kb.KANBAN_SCHEMA_TRIGGERS:
        raw.execute(f"DROP TRIGGER {name}")
    raw.commit()
    raw.close()
    kb._INITIALIZED_PATHS.discard(str(board.resolve()))

    with kbc.connect_closing() as conn:
        names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}

    assert names == kb.KANBAN_SCHEMA_TRIGGERS


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT OR REPLACE INTO tasks (id, title, status, created_at) VALUES (?, 'HIJACKED', 'running', 0)",
        "REPLACE INTO tasks (id, title, status, created_at) VALUES (?, 'HIJACKED', 'running', 0)",
        "INSERT INTO tasks (id, title, status, created_at) VALUES (?, 'HIJACKED', 'running', 0) "
        "ON CONFLICT(id) DO UPDATE SET title = excluded.title, status = excluded.status",
        "INSERT OR IGNORE INTO tasks (id, title, status, created_at) VALUES (?, 'HIJACKED', 'running', 0)",
    ],
    ids=["or-replace", "replace", "upsert", "or-ignore"],
)
def test_foreign_writer_cannot_insert_over_an_existing_task(board, statement):
    """``INSERT OR REPLACE`` drops the old row by implicit delete + insert: the
    id is well-formed and no UPDATE runs, so the shape and immutability guards
    never fire. Every insert onto an existing id is refused, and the claimed
    row keeps its assignee and claim."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="alpha", assignee="coder")
        other = kb.create_task(conn, title="beta", assignee="coder")
        assert kb.claim_task(conn, tid) is not None
        before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    raw = sqlite3.connect(board, isolation_level=None)
    with pytest.raises(sqlite3.IntegrityError, match="insert over an existing task id"):
        raw.execute(statement, (tid,))
    raw.close()

    with kbc.connect_closing() as conn:
        after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
    assert after == before
    assert after["claim_lock"] and after["assignee"] == "coder"
    assert _ids(board) == {tid, other}


def test_create_task_id_collision_retry_still_works(board, monkeypatch):
    """The replace guard surfaces as IntegrityError, which create_task's
    one-shot collision retry already catches: a colliding mint still lands on
    a fresh id instead of failing the create."""
    with kbc.connect_closing() as conn:
        existing = kb.create_task(conn, title="first", assignee="coder")
        minted = iter([existing, "t_00c0ffee"])
        monkeypatch.setattr(kb, "_new_task_id", lambda: next(minted))
        assert kb.create_task(conn, title="second", assignee="coder") == "t_00c0ffee"
    assert _ids(board) == {existing, "t_00c0ffee"}
