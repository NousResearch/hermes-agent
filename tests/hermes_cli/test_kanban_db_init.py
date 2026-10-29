from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


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


def test_connect_reinitializes_schema_when_db_file_vanished(tmp_path, monkeypatch):
    """#83445: the schema cache is process-local, but the schema is on disk.

    A long-lived process (gateway, dispatcher, dashboard API) that already
    initialized a path keeps taking the ``_INITIALIZED_PATHS`` fast path after
    the file is deleted underneath it. SQLite recreates an empty DB on the next
    open, so every query then fails with ``no such table: tasks`` and the board
    renders empty until that process itself is restarted.
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

    with kbc.connect_closing(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    assert "tasks" in _tables(db_path)


def test_connect_reinitializes_schema_when_db_replaced_by_empty_file(tmp_path, monkeypatch):
    """Same defect, restore shape: the file still exists and passes both the
    header and the integrity probes, but carries no schema at all."""
    db_path = _default_board_db(tmp_path, monkeypatch)

    with kbc.connect_closing(db_path):
        pass

    for suffix in ("", "-wal", "-shm"):
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
    sqlite3.connect(str(db_path)).close()
    assert "tasks" not in _tables(db_path)

    with kbc.connect_closing(db_path) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) VALUES ('t-2', 'T', 'ready', 1000)"
        )
        conn.commit()
    assert "tasks" in _tables(db_path)


def test_healthy_fast_path_stays_lock_free(tmp_path, monkeypatch):
    """The self-heal must cost nothing in steady state: an intact cached path
    still skips the cross-process init lock (#36644), and only pays for it when
    the schema is actually gone."""
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
    with kbc.connect_closing(db_path):
        pass
    assert len(locks) == 1


# --------------------------------------------------------------------------- #
# GOV-F25.c AC-4 — durable guard (subscribe + re-subscribe) + migration/rebuild + inheritance #
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402
from hermes_cli import kanban_db_notify as kbn  # noqa: E402,F401
from tests.gov_f25c_support import (  # noqa: E402
    _seed_task, _notify_sub, _notify_sub_columns, _default_retry_policy, _default_pending_event_id,
    _db_without_optional_columns, _drifted_db_with_durable_row, _sub_retry_policy, _sub_pending_event_id,
)


@pytest.fixture
def kanban_conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    try:
        yield conn
    finally:
        conn.close()


def test_ac_gov_f25_c_4(kanban_conn):
    """add_notify_sub rejects durable on push AND on a non-wake-capable mode (subscribe + re-subscribe); retry_policy + pending_event_id survive migration/rebuild; child inherits durable."""
    from hermes_cli import kanban_db_notify, kanban_db_connect

    task_id = _seed_task(kanban_conn)
    with pytest.raises(ValueError):
        kanban_db_notify.add_notify_sub(
            kanban_conn, task_id=task_id, platform="telegram", chat_id="c1", retry_policy="durable")
    with pytest.raises(ValueError):
        kanban_db_notify.add_notify_sub(
            kanban_conn, task_id=task_id, platform="api_server", chat_id="c2",
            delivery_mode="notify", retry_policy="durable")
    kanban_db_notify.add_notify_sub(
        kanban_conn, task_id=task_id, platform="api_server", chat_id="c3",
        delivery_mode="wake", retry_policy="durable")
    with pytest.raises(ValueError):
        kanban_db_notify.add_notify_sub(
            kanban_conn, task_id=task_id, platform="api_server", chat_id="c3",
            delivery_mode="notify", retry_policy="durable")
    assert _notify_sub(kanban_conn, task_id=task_id, chat_id="c3").delivery_mode == "wake"
    with pytest.raises(ValueError):
        kanban_db_notify.add_notify_sub(
            kanban_conn, task_id=task_id, platform="api_server", chat_id="c3", delivery_mode="notify")
    assert _notify_sub(kanban_conn, task_id=task_id, chat_id="c3").delivery_mode == "wake"

    migrate_path = _db_without_optional_columns()
    kanban_db_connect.init_db(db_path=migrate_path)
    with kanban_db_connect.connect_closing(db_path=migrate_path) as migrated:
        cols = _notify_sub_columns(migrated)
        assert "retry_policy" in cols and "pending_event_id" in cols
        assert _default_retry_policy(migrated) == "default"
        assert _default_pending_event_id(migrated) is None
    rebuild_path = _drifted_db_with_durable_row(chat_id="pre", pending_event_id=7)
    kanban_db_connect.init_db(db_path=rebuild_path)
    with kanban_db_connect.connect_closing(db_path=rebuild_path) as rebuilt:
        assert _sub_retry_policy(rebuilt, chat_id="pre") == "durable"
        assert _sub_pending_event_id(rebuilt, chat_id="pre") == 7

    from hermes_cli.kanban_db import _inherit_notify_subs
    child_task = _seed_task(kanban_conn)
    _inherit_notify_subs(kanban_conn, child_task, (task_id,))
    inherited = _notify_sub(kanban_conn, task_id=child_task, chat_id="c3")
    assert inherited.retry_policy == "durable"
