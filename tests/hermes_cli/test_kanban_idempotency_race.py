"""Concurrency regression: ``create_task`` idempotency-key race-safety.

``idx_tasks_idempotency`` used to be a plain (non-UNIQUE) index, so two
connections that both SELECT-miss on the same key could each INSERT, leaving
two active task rows for one idempotency key. The UNIQUE partial index over
active (non-archived) rows plus IntegrityError recovery in ``create_task``
make the losing writer return the winner's id instead of inserting a duplicate.
"""

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty, migrated kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_task_idempotency_key_is_race_safe(kanban_home, monkeypatch):
    """Two concurrent create_task calls with one idempotency_key -> one row.

    The barrier is placed *inside* create_task — between its fast-path
    idempotency SELECT and its INSERT — rather than before the call. A barrier
    before create_task can let one caller fully commit before the other even
    runs its SELECT, so the other returns via the fast-path lookup and the
    UNIQUE-index + IntegrityError-recovery path is never exercised (the test
    passes without proving anything). ``_new_task_id`` runs after the fast-path
    SELECT and before ``write_txn``'s INSERT, so gating it there synchronises
    both writers in exactly the race window: both have SELECT-missed before
    either INSERTs.
    """
    key = "dod:fix:v1:race-safe-check"
    results: dict = {}
    errors: dict = {}

    orig_new_task_id = kb._new_task_id
    gate = threading.Barrier(2)
    gated: set = set()
    gate_lock = threading.Lock()

    def gated_new_task_id():
        task_id = orig_new_task_id()
        ident = threading.get_ident()
        with gate_lock:
            first = ident not in gated
            if first:
                gated.add(ident)
        # Only sync on the first call per thread (the id-collision retry loop
        # could call again; the loser recovers before that anyway).
        if first:
            gate.wait(timeout=10)
        return task_id

    monkeypatch.setattr(kb, "_new_task_id", gated_new_task_id)

    def worker(n: int) -> None:
        try:
            with kb.connect() as conn:
                results[n] = kb.create_task(
                    conn, title="racer-%d" % n, idempotency_key=key
                )
        except Exception as exc:  # noqa: BLE001 - surfaced via assert below
            errors[n] = repr(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, "create_task raised under concurrency: %r" % errors
    # Both callers resolve to the SAME task id (loser recovers the winner's row).
    assert results[0] == results[1], "racers got different ids: %r" % results
    # Exactly one active (non-archived) row carries the key.
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived'",
            (key,),
        ).fetchall()
    assert len(rows) == 1, "expected exactly 1 active row for key, got %d" % len(rows)


def test_empty_idempotency_key_is_not_unique(kanban_home):
    """An empty key means "no idempotency" — repeated creates must not collide.

    The UNIQUE partial index would otherwise treat a stored ''- like any other
    key and reject the second keyless create with an IntegrityError. create_task
    normalises '' (and whitespace-only) to NULL, and the index excludes both
    NULL and '', so keyless creates always succeed.
    """
    with kb.connect() as conn:
        # Two identical empty keys is the reported regression: pre-fix the second
        # stored-'' INSERT hits the UNIQUE index and raises IntegrityError.
        a = kb.create_task(conn, title="keyless-a", idempotency_key="")
        b = kb.create_task(conn, title="keyless-b", idempotency_key="")
        c = kb.create_task(conn, title="keyless-c", idempotency_key="   ")
        d = kb.create_task(conn, title="keyless-d", idempotency_key=None)

    assert len({a, b, c, d}) == 4, "keyless creates collided: %r" % [a, b, c, d]
    with kb.connect() as conn:
        # Empty/whitespace keys are normalised to NULL, never stored as ''.
        stored = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = ''"
        ).fetchone()["n"]
    assert stored == 0, "empty key was stored instead of normalised to NULL"


def test_migration_dedupes_preexisting_active_duplicates(kanban_home):
    """A pre-fix DB with duplicate active keys migrates cleanly.

    Before this fix, ``idx_tasks_idempotency`` was non-unique, so a board could
    already hold two active rows for one key. Building the UNIQUE index directly
    over that dirty data would raise and make the DB unopenable. The migration
    must de-dupe first (keep newest per the create_task fast-path order, archive
    the rest) and only then build the unique index.
    """
    key = "dod:fix:v1:legacy-dupe"
    path = kb.kanban_db_path()

    # Reproduce the pre-fix on-disk state: restore a plain index and insert two
    # active rows sharing one key (create_task can no longer produce this).
    with kb.connect() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_tasks_idempotency")
        conn.execute("CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)")
        for i, created_at in ((0, 100), (1, 200)):  # older, then newer
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at, idempotency_key) "
                "VALUES (?, ?, 'ready', ?, ?)",
                ("dupe-%d" % i, "legacy-%d" % i, created_at, key),
            )

    # Re-run the migration pass, as opening an upgraded DB would.
    kb.init_db(path)

    with kb.connect() as conn:
        active = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
            (key,),
        ).fetchall()
        assert [r["id"] for r in active] == ["dupe-1"], (
            "expected only the newest row to stay active, got %r"
            % [r["id"] for r in active]
        )
        archived = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? AND status = 'archived'",
            (key,),
        ).fetchall()
        assert [r["id"] for r in archived] == ["dupe-0"], (
            "expected the older duplicate to be archived, got %r"
            % [r["id"] for r in archived]
        )
        # The unique index is now in force: a fresh active duplicate is rejected.
        with pytest.raises(sqlite3.IntegrityError):
            with kb.write_txn(conn):
                conn.execute(
                    "INSERT INTO tasks (id, title, status, created_at, idempotency_key) "
                    "VALUES ('dupe-2', 'legacy-2', 'ready', 300, ?)",
                    (key,),
                )
