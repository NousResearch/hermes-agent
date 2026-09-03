"""Race convergence for idempotent create_task (t_e548b1d2 slice of t_ec4b92fd).

With the UNIQUE idx_tasks_idempotency (PR #95742) the historical
"race is acceptable" comment is no longer true in the good direction:
two concurrent same-key creators no longer both insert — instead the
loser now RAISES sqlite3.IntegrityError (fail-loud interim behavior,
documented in #95742). This slice makes racing creators CONVERGE:
exactly one row persists and every caller gets that row's id.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def race_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (fresh = UNIQUE index)."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_race_loser_converges_on_existing_row(race_home):
    """Deterministic loser: pre-existing archived row with the same key.

    Exercises the exact interleaving the pre-check cannot see: the row is
    ARCHIVED (so the pre-check's historical status filter missed it) and the
    UNIQUE index turns the would-be duplicate INSERT into IntegrityError.
    create_task must converge on the archived row's id — admitted-once
    means admitted-forever — instead of raising or duplicating.
    """
    key = "race-archived-key"
    with kb.connect() as conn:
        # Raw-SQL archived row: what a legacy estate looked like pre-purge.
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, idempotency_key) "
            "VALUES ('t_arch_1', 'archived original', 'archived', 1, ?)",
            (key,),
        )
        conn.commit()

        got = kb.create_task(conn, title="racing creator", idempotency_key=key)

        assert got == "t_arch_1", (
            "create_task racing an archived same-key row must return that "
            "row's id (converge), not raise and not insert a duplicate"
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (key,)
        ).fetchone()[0]
        assert count == 1, f"expected exactly 1 row for key, found {count}"


def test_empty_string_key_means_no_key(race_home):
    """Regression: '' key must behave identically to None.

    The guard historically treats '' as falsy (no idempotency), but the
    INSERT stored '' as a value — under the UNIQUE index a second ''-key
    create then crashed with IntegrityError ('' collided with the stored
    ''). Normalizing '' to NULL at entry makes guard and storage agree:
    each ''-key create is an ordinary independent task.
    """
    a = kb.create_task(kb.connect(), title="empty one", idempotency_key="")
    b = kb.create_task(kb.connect(), title="empty two", idempotency_key="")
    assert a != b, "'' key must not be idempotent — it means no key"
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ''"
        ).fetchone()[0]
    assert rows == 0, f"'' must be normalized to NULL, found {rows} stored '' keys"


def test_concurrent_same_key_creates_produce_exactly_one_row(race_home):
    """Real concurrency: 8 threads, separate connections, one key.

    Every thread calls create_task with the same idempotency_key. Exactly
    one row must persist and every caller must receive that row's id.
    """
    key = "race-concurrent-key"
    n = 8

    def create_one():
        conn = kb.connect()
        try:
            return kb.create_task(conn, title="concurrent creator", idempotency_key=key)
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        ids = list(pool.map(lambda _: create_one(), range(n)))

    assert len(set(ids)) == 1, f"callers diverged on id: {ids}"
    with kb.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", (key,)
        ).fetchone()[0]
        assert count == 1, f"expected exactly 1 row for key, found {count}"
        assert conn.execute("SELECT id FROM tasks WHERE idempotency_key = ?", (key,)).fetchone()[
            0
        ] == ids[0]
