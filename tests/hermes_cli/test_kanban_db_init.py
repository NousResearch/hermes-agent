from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_writer


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


def test_unactivated_callers_bootstrap_missing_db_under_concurrency(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            conn = kb.connect(board="default")
            conn.close()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert db_path.exists()
    with kb.connect(board="default") as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "max_retries" in cols

    service = kanban_writer.KanbanWriterService(db_path)
    service.start()
    service.stop()
    with kb.connect(board="default") as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "max_retries" in cols


def test_legacy_text_pk_tables_rebuilt_to_integer_autoincrement(tmp_path, monkeypatch):
    """A pre-AUTOINCREMENT DB is migrated in place: id columns become INTEGER
    PKs, ``last_event_id`` becomes INTEGER, data is preserved, and indexes
    are recreated (DROP TABLE would otherwise take them down)."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path) as conn:
        for table in ("task_events", "task_comments", "task_runs"):
            id_col = {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")}["id"]
            assert id_col["type"].upper() == "INTEGER" and id_col["pk"] == 1

        lei = {r["name"]: r for r in conn.execute("PRAGMA table_info(kanban_notify_subs)")}
        assert lei["last_event_id"]["type"].upper() == "INTEGER"

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


def test_rebuilt_schema_matches_fresh_db(tmp_path, monkeypatch):
    """The rebuilt tables must be structurally identical to a fresh DB, so the
    hand-written DDL in ``_REBUILD_SPECS`` can't silently drift from SCHEMA_SQL."""
    legacy_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(legacy_path)
    fresh_path = kb.kanban_db_path(board="fresh")
    fresh_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(fresh_path.resolve()))

    with kb.connect(legacy_path) as migrated, kb.connect(fresh_path) as fresh:
        for table in ("task_events", "task_comments", "task_runs", "kanban_notify_subs"):
            assert _table_struct(migrated, table) == _table_struct(fresh, table)


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Re-opening an already-migrated DB is a no-op and leaves data intact."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path):
        pass
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        id_col = {r["name"]: r for r in conn.execute("PRAGMA table_info(task_events)")}["id"]
        assert id_col["type"].upper() == "INTEGER"
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2


def test_unseen_events_for_sub_survives_migrated_db(tmp_path, monkeypatch):
    """The crash that motivated #35096 — ``int(None)`` on a NULL cursor — is
    gone after migration; the notifier query returns an integer cursor."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path) as conn:
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id="task-1", platform="telegram", chat_id="123"
        )
        assert isinstance(cursor, int)
        assert isinstance(events, list)


def test_numeric_legacy_notify_cursor_resets_after_event_id_rebuild(
    tmp_path, monkeypatch
):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE kanban_notify_subs SET last_event_id='100'")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        stored_cursor = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs"
        ).fetchone()["last_event_id"]
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id="task-1", platform="telegram", chat_id="123"
        )

    assert stored_cursor == 0
    assert len(events) == 2
    assert cursor == max(event.id for event in events)


def test_event_rebuild_resets_cursor_when_notify_schema_is_already_current(
    tmp_path, monkeypatch
):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)
    create_sql, index_sqls = kb._REBUILD_SPECS["kanban_notify_subs"]
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE kanban_notify_subs")
        conn.execute(create_sql)
        for index_sql in index_sqls:
            conn.execute(index_sql)
        conn.execute(
            "INSERT INTO kanban_notify_subs "
            "(task_id, platform, chat_id, created_at, last_event_id) "
            "VALUES ('task-1', 'telegram', '123', 1000, 100)"
        )
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        stored_cursor = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs"
        ).fetchone()["last_event_id"]
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id="task-1", platform="telegram", chat_id="123"
        )

    assert stored_cursor == 0
    assert len(events) == 2
    assert cursor == max(event.id for event in events)


def test_notify_only_rebuild_preserves_numeric_cursor(tmp_path, monkeypatch):
    db_path = _setup_home(tmp_path, monkeypatch)
    with kb.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) "
            "VALUES ('task-1', 'T', 'done', 1000)"
        )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE kanban_notify_subs")
        conn.execute(
            "CREATE TABLE kanban_notify_subs ("
            "task_id TEXT NOT NULL, platform TEXT NOT NULL, "
            "chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', "
            "user_id TEXT, created_at INTEGER NOT NULL, last_event_id TEXT, "
            "PRIMARY KEY (task_id, platform, chat_id, thread_id))"
        )
        conn.execute(
            "INSERT INTO kanban_notify_subs "
            "(task_id, platform, chat_id, created_at, last_event_id) "
            "VALUES ('task-1', 'telegram', '123', 1000, '100')"
        )
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs"
        ).fetchone()["last_event_id"]

    assert cursor == 100


def test_legacy_running_task_is_backfilled_after_run_table_rebuild(
    tmp_path, monkeypatch
):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE task_runs")
        conn.execute(
            "CREATE TABLE task_runs ("
            "id TEXT PRIMARY KEY NOT NULL, task_id TEXT NOT NULL, "
            "profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL)"
        )
        conn.execute(
            "UPDATE tasks SET status='running', assignee='ava', "
            "claim_lock='legacy-claim', claim_expires=9999, worker_pid=42, "
            "started_at=1234, current_run_id=NULL WHERE id='task-1'"
        )
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        task = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id='task-1'"
        ).fetchone()
        run = conn.execute(
            "SELECT id, claim_lock, worker_pid FROM task_runs WHERE task_id='task-1'"
        ).fetchone()

    assert isinstance(task["current_run_id"], int)
    assert task["current_run_id"] == run["id"]
    assert run["claim_lock"] == "legacy-claim"
    assert run["worker_pid"] == 42


def test_task_run_rebuild_remaps_cross_table_run_references(tmp_path, monkeypatch):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE task_events ADD COLUMN run_id TEXT")
        conn.execute("UPDATE task_runs SET id='run-z', profile='target' WHERE id='r-1'")
        conn.execute(
            "INSERT INTO task_runs VALUES ('run-a', 'task-1', 'other', 'done', 900)"
        )
        conn.execute("UPDATE tasks SET current_run_id='run-z' WHERE id='task-1'")
        conn.execute("UPDATE task_events SET run_id='run-z' WHERE id='e-1'")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        target_run_id = conn.execute(
            "SELECT id FROM task_runs WHERE profile='target'"
        ).fetchone()["id"]
        task_run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id='task-1'"
        ).fetchone()["current_run_id"]
        event_run_id = conn.execute(
            "SELECT run_id FROM task_events WHERE kind='completed'"
        ).fetchone()["run_id"]

    assert isinstance(target_run_id, int)
    assert task_run_id == target_run_id
    assert event_run_id == target_run_id


def test_populated_legacy_writer_ledger_gains_attribution_without_row_loss(
    tmp_path, monkeypatch
):
    db_path = _setup_home(tmp_path, monkeypatch)
    with kb.connect(db_path):
        pass
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            DROP TABLE kanban_writer_requests;
            CREATE TABLE kanban_writer_requests (
                request_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            INSERT INTO kanban_writer_requests
                (request_id, operation, payload_digest, response, created_at)
            VALUES ('legacy-request', 'create_task', 'abc', '{"ok":true}', 1);
            """
        )
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        columns = {row["name"] for row in conn.execute(
            "PRAGMA table_info(kanban_writer_requests)"
        )}
        row = conn.execute(
            "SELECT request_id, board, actor_profile, source, writer_pid, operation "
            "FROM kanban_writer_requests WHERE request_id='legacy-request'"
        ).fetchone()

    assert {"board", "actor_profile", "source", "writer_pid"} <= columns
    assert tuple(row) == (
        "legacy-request",
        "",
        "",
        "",
        0,
        "create_task",
    )
