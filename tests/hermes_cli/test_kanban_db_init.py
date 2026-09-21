from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


def _make_legacy_db(path: Path) -> None:
    """Write a kanban DB with the pre-AUTOINCREMENT (TEXT PK) schema for the
    four tables #35096 affects, keeping every other table current so the
    additive-column migration runs cleanly on top.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript(
        """
        DROP TABLE task_events;
        DROP TABLE task_comments;
        DROP TABLE task_runs;
        DROP TABLE kanban_notify_subs;
        CREATE TABLE task_comments (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            author TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL);
        CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL);
        CREATE TABLE task_runs (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL);
        CREATE TABLE kanban_notify_subs (task_id TEXT NOT NULL, platform TEXT NOT NULL,
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,
            created_at INTEGER NOT NULL, last_event_id TEXT,
            PRIMARY KEY (task_id, platform, chat_id, thread_id));
        """
    )
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('task-1', 'T', 'done', 1000)")
    conn.execute("INSERT INTO task_comments VALUES ('c-1', 'task-1', 'agent', 'hi', 1500)")
    conn.execute("INSERT INTO task_events VALUES ('e-1', 'task-1', 'completed', NULL, 2000)")
    conn.execute("INSERT INTO task_events VALUES ('e-2', 'task-1', 'blocked', NULL, 2100)")
    conn.execute("INSERT INTO task_runs VALUES ('r-1', 'task-1', 'default', 'done', 1000)")
    conn.execute(
        "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, created_at, last_event_id) "
        "VALUES ('task-1', 'telegram', '123', 1000, 'e-1')"
    )
    conn.commit()
    conn.close()


def _setup_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="legacy")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return db_path


def _table_struct(conn: sqlite3.Connection, table: str):
    cols = [
        (r["name"], (r["type"] or "").upper(), r["notnull"], r["pk"])
        for r in conn.execute(f"PRAGMA table_info({table})")
    ]
    idx = sorted(
        r["name"]
        for r in conn.execute(f"PRAGMA index_list({table})")
        if not r["name"].startswith("sqlite_")
    )
    return cols, idx




def test_legacy_text_pk_tables_rebuilt_to_integer_autoincrement(tmp_path, monkeypatch):
    """A pre-AUTOINCREMENT DB is migrated in place: id columns become INTEGER
    PKs, ``last_event_id`` becomes INTEGER, data is preserved, and indexes
    are recreated (DROP TABLE would otherwise take them down)."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kbc.connect(db_path) as conn:
        for table in ("task_events", "task_comments", "task_runs"):
            id_col = {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")}["id"]
            assert id_col["type"].upper() == "INTEGER" and id_col["pk"] == 1

        lei = {r["name"]: r for r in conn.execute("PRAGMA table_info(kanban_notify_subs)")}
        assert lei["last_event_id"]["type"].upper() == "INTEGER"
        assert "delivery_metadata" in lei

        # Data preserved across the rebuild.
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2
        assert conn.execute("SELECT body FROM task_comments").fetchone()["body"] == "hi"
        assert len(conn.execute("SELECT * FROM task_runs").fetchall()) == 1
        # Non-numeric legacy cursor ("e-1") casts to 0.
        assert conn.execute("SELECT last_event_id FROM kanban_notify_subs").fetchone()["last_event_id"] == 0

        # Indexes restored, including idx_events_run (added by the additive pass).
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        for name in ("idx_events_task", "idx_events_run", "idx_comments_task",
                     "idx_runs_task", "idx_runs_status", "idx_notify_task"):
            assert name in indexes

        # AUTOINCREMENT actually works after the rebuild.
        conn.execute("INSERT INTO task_events (task_id, kind, created_at) VALUES ('task-1', 'completed', 3000)")
        new_id = conn.execute("SELECT id FROM task_events ORDER BY id DESC LIMIT 1").fetchone()["id"]
        assert isinstance(new_id, int) and new_id >= 1




def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Re-opening an already-migrated DB is a no-op and leaves data intact."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kbc.connect(db_path):
        pass
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kbc.connect(db_path) as conn:
        id_col = {r["name"]: r for r in conn.execute("PRAGMA table_info(task_events)")}["id"]
        assert id_col["type"].upper() == "INTEGER"
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2


def test_unseen_events_for_sub_survives_migrated_db(tmp_path, monkeypatch):
    """The crash that motivated #35096 — ``int(None)`` on a NULL cursor — is
    gone after migration; the notifier query returns an integer cursor."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kbc.connect(db_path) as conn:
        cursor, events = kbn.unseen_events_for_sub(
            conn, task_id="task-1", platform="telegram", chat_id="123"
        )
        assert isinstance(cursor, int)
        assert isinstance(events, list)


def _default_board_db(tmp_path, monkeypatch) -> Path:
    """Point the kanban root at a temp home and return the default board's DB
    (the back-compat top-level ``<root>/kanban.db`` #83445 reports on)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return db_path


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def test_connect_refuses_when_db_file_vanished(tmp_path, monkeypatch):
    """A vanished board file is NOT a fresh board (#83445 follow-up).

    The schema cache is process-local, but the schema is on disk. A long-lived
    process (gateway, dispatcher, dashboard API) that already initialized a path
    keeps taking the ``_INITIALIZED_PATHS`` fast path after the file is deleted
    underneath it. Re-running the schema script there recreated an EMPTY board:
    the cards were already gone, and the board came back looking alive with
    nothing on it, so the loss read as "my tasks disappeared" instead of an
    error. Refuse, and leave nothing behind for the next reader to mistake for a
    live board.
    """
    db_path = _default_board_db(tmp_path, monkeypatch)

    with kbc.connect_closing(db_path) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) VALUES ('t-1', 'T', 'ready', 1000)"
        )
        conn.commit()
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

    # External deletion (manual cleanup, restore, sync tool) while the process
    # that cached this path is still alive.
    for suffix in ("", "-wal", "-shm"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)

    with pytest.raises(kbc.KanbanDbReplacedError):
        with kbc.connect_closing(db_path):
            pass
    # The refusal is not a tidy-up: no empty board is minted on the way out.
    assert not db_path.exists()


def test_connect_refuses_when_db_replaced_by_empty_file(tmp_path, monkeypatch):
    """Same defect, restore shape: the file exists and passes the header probe,
    but carries no schema at all. Re-running the schema script on it would hand
    back an empty board — so it is refused, and the replacement's bytes are left
    exactly as they were (they are the evidence)."""
    db_path = _default_board_db(tmp_path, monkeypatch)

    with kbc.connect_closing(db_path):
        pass

    for suffix in ("", "-wal", "-shm"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
    db_path.write_bytes(b"")

    with pytest.raises(kbc.KanbanDbReplacedError):
        with kbc.connect_closing(db_path):
            pass
    # Untouched: the replacement's bytes are the evidence, and no empty board is
    # grafted on top of them.
    assert db_path.stat().st_size == 0
    assert "tasks" not in _tables(db_path)


def test_healthy_fast_path_stays_lock_free(tmp_path, monkeypatch):
    """The check must cost nothing in steady state: an intact cached path still
    skips the cross-process init lock (#36644), and a board that is gone is
    refused before the lock — nothing is written, so nothing needs serializing."""
    db_path = _default_board_db(tmp_path, monkeypatch)

    with kbc.connect_closing(db_path):
        pass

    locks: list[Path] = []
    real_lock = kbc._cross_process_init_lock

    @contextlib.contextmanager
    def recording_lock(path):
        locks.append(path)
        with real_lock(path):
            yield

    monkeypatch.setattr(kbc, "_cross_process_init_lock", recording_lock)

    with kbc.connect_closing(db_path):
        pass
    assert locks == []

    db_path.unlink()
    with pytest.raises(kbc.KanbanDbReplacedError):
        with kbc.connect_closing(db_path):
            pass
    assert locks == []
