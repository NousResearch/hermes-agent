"""Idempotency must be an IDENTITY, not a hint.

`create_task(idempotency_key=...)` documents "an existing non-archived task with the key is
returned instead of creating a duplicate", and the launcher's whole submission contract rests
on that (one request ⇒ one board task). The check runs *outside* the write transaction, so
concurrent creators can both pass it and both INSERT.

These tests pin the invariant rather than the implementation: exactly one non-archived task
exists for a key, every caller observes the same id, and a board that predates the constraint
still opens.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    return home


def _rows_for_key(db_path, key: str) -> list[str]:
    conn = kbc.connect(db_path)
    try:
        return [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM tasks WHERE idempotency_key = ? ORDER BY created_at", (key,)
            )
        ]
    finally:
        conn.close()


def test_concurrent_same_key_creates_exactly_one_task(kanban_home, monkeypatch):
    """N simultaneous creators, one key ⇒ one row and one shared id.

    The interleaving is forced, not hoped for: the first `write_txn` in each thread waits on a
    barrier, so every thread has already executed the pre-transaction idempotency lookup before
    any of them proceeds to INSERT.
    """
    db_path = kb.kanban_db_path()
    key = "race:concurrent:one-key"
    threads_n = 8

    barrier = threading.Barrier(threads_n, timeout=15)
    local = threading.local()
    original = kb.write_txn

    def patched(conn, **kwargs):
        if not getattr(local, "waited", False):
            local.waited = True
            barrier.wait()
        return original(conn, **kwargs)

    monkeypatch.setattr(kb, "write_txn", patched)

    returned: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        conn = kbc.connect(db_path)
        try:
            task_id = kb.create_task(
                conn, title=f"worker {i}", assignee="crypto", idempotency_key=key
            )
            with lock:
                returned.append(task_id)
        except BaseException as exc:  # noqa: BLE001 - surfacing the failure IS the assertion
            with lock:
                errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"no caller may fail: {errors!r}"
    assert len(returned) == threads_n

    rows = _rows_for_key(db_path, key)
    assert len(rows) == 1, f"exactly one task may exist for the key, found {len(rows)}: {rows!r}"
    assert set(returned) == {rows[0]}, (
        "every caller must be handed the SAME id; "
        f"rows={rows!r} returned={sorted(set(returned))!r}"
    )


def test_sequential_same_key_returns_the_same_id(kanban_home):
    """The documented create-or-find path still works for the single-threaded case."""
    db_path = kb.kanban_db_path()
    key = "race:sequential"
    conn = kbc.connect(db_path)
    try:
        first = kb.create_task(conn, title="first", assignee="a", idempotency_key=key)
        second = kb.create_task(conn, title="second", assignee="a", idempotency_key=key)
    finally:
        conn.close()

    assert first == second
    assert len(_rows_for_key(db_path, key)) == 1


def test_archived_task_does_not_block_a_new_create_with_the_same_key(kanban_home):
    """The constraint is PARTIAL: archiving releases the key, as create-or-find documents.

    A plain unique index would make an archived task hold the key forever and make re-creating
    after an archive impossible.
    """
    db_path = kb.kanban_db_path()
    key = "race:archived"
    conn = kbc.connect(db_path)
    try:
        original_id = kb.create_task(conn, title="archived", assignee="a", idempotency_key=key)
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (original_id,))
        conn.commit()
        replacement_id = kb.create_task(conn, title="replacement", assignee="a", idempotency_key=key)
    finally:
        conn.close()

    assert replacement_id != original_id
    rows = _rows_for_key(db_path, key)
    assert len(rows) == 2, "one archived + one live row is the documented behaviour"
    assert rows[-1] == replacement_id


def test_legacy_board_with_duplicates_still_opens(kanban_home):
    """A board that ALREADY holds duplicates must remain usable.

    SQLite refuses to build a unique index over existing duplicates. If the migration let that
    raise, every affected board would become unopenable — strictly worse than the race it is
    fixing. The migration must degrade instead: keep the plain lookup index and leave the
    duplicates for an operator to resolve.
    """
    db_path = kb.kanban_db_path()
    key = "race:legacy-duplicates"

    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("DROP INDEX IF EXISTS idx_tasks_idempotency")
        for i in range(2):
            raw.execute(
                "INSERT INTO tasks (id, title, status, created_at, idempotency_key) "
                "VALUES (?, ?, 'running', ?, ?)",
                (f"t_legacy{i}", f"legacy {i}", 1000 + i, key),
            )
        raw.commit()
    finally:
        raw.close()

    # Re-opening the board runs the migration. Drop this process's "already
    # initialized" cache entry so connect() genuinely re-runs it, which is what a
    # fresh process (gateway restart, CLI invocation) does.
    kbc._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    conn = kbc.connect(db_path)
    try:
        indexes = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert "idx_tasks_idempotency" in indexes, (
            "the plain lookup index must exist even when the unique one cannot be built"
        )
        # And the board is still usable for unrelated creates.
        created = kb.create_task(conn, title="after legacy open", assignee="a")
        assert created
    finally:
        conn.close()