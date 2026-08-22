"""Regression coverage for Kanban task idempotency-key races and migration."""

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Use an isolated, initialized Kanban database."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_task_idempotency_key_is_race_safe(kanban_home, monkeypatch):
    """Three connections racing after the lookup return one task id."""
    key = "dod:fix:v1:race-safe-check"
    results: dict[int, str] = {}
    errors: dict[int, str] = {}
    gate = threading.Barrier(3)
    gated_threads: set[int] = set()
    gate_lock = threading.Lock()
    original_new_task_id = kb._new_task_id

    def gated_new_task_id() -> str:
        task_id = original_new_task_id()
        thread_id = threading.get_ident()
        with gate_lock:
            first_call = thread_id not in gated_threads
            if first_call:
                gated_threads.add(thread_id)
        # Synchronize after every caller has missed the fast-path SELECT, but
        # before any caller enters write_txn. Retries do not re-enter the gate.
        if first_call:
            gate.wait(timeout=10)
        return task_id

    monkeypatch.setattr(kb, "_new_task_id", gated_new_task_id)

    def worker(number: int) -> None:
        try:
            with kb.connect_closing() as conn:
                results[number] = kb.create_task(
                    conn, title=f"racer-{number}", idempotency_key=key
                )
        except Exception as exc:  # noqa: BLE001 - surfaced by assertions below
            errors[number] = repr(exc)

    threads = [threading.Thread(target=worker, args=(number,)) for number in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not any(thread.is_alive() for thread in threads), "race test deadlocked"
    assert not errors, f"create_task raised under concurrency: {errors!r}"
    assert results.keys() == {0, 1, 2}
    assert len(set(results.values())) == 1, f"racers got different ids: {results!r}"

    with kb.connect_closing() as conn:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived'",
            (key,),
        ).fetchall()
    assert len(rows) == 1, f"expected one active row for key, got {len(rows)}"


def test_empty_and_whitespace_idempotency_keys_are_absent(kanban_home):
    """Empty keys do not turn otherwise independent creates into duplicates."""
    with kb.connect_closing() as conn:
        task_ids = [
            kb.create_task(conn, title="empty", idempotency_key=""),
            kb.create_task(conn, title="whitespace", idempotency_key="   "),
            kb.create_task(conn, title="none", idempotency_key=None),
        ]
        stored = conn.execute(
            "SELECT idempotency_key FROM tasks ORDER BY created_at, id"
        ).fetchall()

    assert len(set(task_ids)) == 3
    assert [row["idempotency_key"] for row in stored] == [None, None, None]


def test_idempotency_key_can_be_reused_after_archive(kanban_home):
    """Archiving releases a key while retaining the archived task row."""
    key = "dod:fix:v1:archive-reuse"
    with kb.connect_closing() as conn:
        archived_id = kb.create_task(conn, title="first", idempotency_key=key)
        assert kb.archive_task(conn, archived_id)
        active_id = kb.create_task(conn, title="second", idempotency_key=key)

        rows = conn.execute(
            "SELECT id, status FROM tasks WHERE idempotency_key = ? ORDER BY id",
            (key,),
        ).fetchall()

    assert active_id != archived_id
    assert {row["status"] for row in rows} == {"archived", "ready"}


def test_migration_archives_duplicate_active_keys_without_losing_history(kanban_home):
    """Dirty pre-fix boards migrate with durable archive evidence and history."""
    key = "dod:fix:v1:legacy-dupe"
    path = kb.kanban_db_path()

    with kb.connect_closing() as conn:
        # Recreate the pre-fix schema state: the old index allowed duplicates.
        conn.execute("DROP INDEX IF EXISTS idx_tasks_idempotency")
        conn.execute("CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)")
        for task_id, title, created_at in (
            ("dupe-old", "legacy-old", 100),
            ("dupe-new", "legacy-new", 200),
        ):
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, status, created_at, idempotency_key) "
                "VALUES (?, ?, 'ready', ?, ?)",
                (task_id, title, created_at, key),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'legacy_history', ?, ?)",
                (task_id, '{"source":"before-migration"}', created_at + 1),
            )

    kb.init_db(path)

    with kb.connect_closing() as conn:
        active = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived'",
            (key,),
        ).fetchall()
        archived = conn.execute(
            "SELECT id, status, claim_lock, claim_expires, worker_pid "
            "FROM tasks WHERE id = 'dupe-old'"
        ).fetchone()
        old_events = kb.list_events(conn, "dupe-old")
        new_events = kb.list_events(conn, "dupe-new")
        index = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_tasks_idempotency'"
        ).fetchone()["sql"]
        with pytest.raises(sqlite3.IntegrityError):
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, status, created_at, idempotency_key) "
                    "VALUES ('dupe-attempt', 'legacy-attempt', 'ready', 300, ?)",
                    (key,),
                )

    assert [row["id"] for row in active] == ["dupe-new"]
    assert archived["status"] == "archived"
    assert archived["claim_lock"] is None
    assert archived["claim_expires"] is None
    assert archived["worker_pid"] is None
    assert [event.kind for event in old_events] == ["legacy_history", "archived"]
    assert old_events[0].payload == {"source": "before-migration"}
    assert old_events[1].payload == {
        "reason": "duplicate_idempotency_key_migration",
        "idempotency_key": key,
    }
    assert [event.kind for event in new_events] == ["legacy_history"]
    assert index is not None
    assert "CREATE UNIQUE INDEX" in index.upper()
    assert "status != 'archived'" in index


@pytest.mark.parametrize(
    "legacy_key, canonical_key",
    [
        ("\tkey\t", "key"),
        ("\nkey\r\n", "key"),
    ],
)
def test_legacy_whitespace_keys_share_canonical_active_task(
    kanban_home, legacy_key, canonical_key,
):
    """ASCII tab/newline legacy keys dedupe without rewriting history/text."""
    path = kb.kanban_db_path()

    with kb.connect_closing() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_tasks_idempotency")
        conn.execute("CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)")
        for task_id, created_at, key in (
            ("legacy-old", 100, legacy_key),
            ("canonical-new", 200, canonical_key),
        ):
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, status, created_at, idempotency_key) "
                "VALUES (?, ?, 'ready', ?, ?)",
                (task_id, task_id, created_at, key),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'legacy_history', ?, ?)",
                (task_id, '{"source":"before-migration"}', created_at + 1),
            )

    kb.init_db(path)

    with kb.connect_closing() as conn:
        active = conn.execute(
            "SELECT id FROM tasks "
            f"WHERE {kb._IDEMPOTENCY_KEY_SQL_EXPR} = ? "
            "AND status != 'archived'",
            (canonical_key,),
        ).fetchall()
        archived = conn.execute(
            "SELECT id, idempotency_key, status FROM tasks WHERE id = 'legacy-old'"
        ).fetchone()
        stored_keys = conn.execute(
            "SELECT idempotency_key FROM tasks ORDER BY created_at, id"
        ).fetchall()
        index = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_tasks_idempotency'"
        ).fetchone()["sql"]

        assert (
            kb.create_task(conn, title="replayed", idempotency_key=canonical_key)
            == "canonical-new"
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="index 'idx_tasks_idempotency'",
        ):
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, status, created_at, idempotency_key) "
                    "VALUES ('legacy-attempt', 'attempt', 'ready', 300, ?)",
                    (legacy_key,),
                )
        old_events = kb.list_events(conn, "legacy-old")
        new_events = kb.list_events(conn, "canonical-new")

    assert [row["id"] for row in active] == ["canonical-new"]
    assert archived["status"] == "archived"
    assert archived["idempotency_key"] == legacy_key
    assert [row["idempotency_key"] for row in stored_keys] == [legacy_key, canonical_key]
    assert [event.kind for event in old_events] == ["legacy_history", "archived"]
    assert [event.kind for event in new_events] == ["legacy_history"]
    expected_sql_expression = (
        "TRIM(idempotency_key, char(9) || char(10) || char(11) || "
        "char(12) || char(13) || ' ')"
    )
    assert expected_sql_expression in index


@pytest.mark.parametrize("winner_key", ("\tkey\t", "\nkey\r"))
def test_integrity_error_recovery_uses_canonical_key(
    kanban_home, monkeypatch, winner_key,
):
    """Raced tab/newline legacy keys resolve through the expression index."""
    with kb.connect_closing() as conn, kb.connect_closing() as winner_conn:
        def insert_winner() -> str:
            with kb.write_txn(winner_conn):
                winner_conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, status, created_at, idempotency_key) "
                    "VALUES ('spaced-winner', 'winner', 'ready', 1, ?)",
                    (winner_key,),
                )
            return "losing-id"

        monkeypatch.setattr(kb, "_new_task_id", insert_winner)
        assert (
            kb.create_task(conn, title="raced", idempotency_key="key")
            == "spaced-winner"
        )


def test_unrelated_integrity_error_is_not_misclassified(kanban_home, monkeypatch):
    """A task-id conflict is still raised instead of becoming an idempotent hit."""
    with kb.connect_closing() as conn:
        existing_id = kb.create_task(
            conn, title="existing", idempotency_key="existing-key"
        )
        monkeypatch.setattr(kb, "_new_task_id", lambda: existing_id)

        with pytest.raises(sqlite3.IntegrityError, match="tasks.id"):
            kb.create_task(conn, title="collision", idempotency_key="new-key")
