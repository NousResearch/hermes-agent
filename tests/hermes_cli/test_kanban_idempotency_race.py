"""Concurrency regression: ``create_task`` idempotency-key race-safety.

``idx_tasks_idempotency`` used to be a plain (non-UNIQUE) index, so two
connections that both SELECT-miss on the same key could each INSERT, leaving
two active task rows for one idempotency key. The UNIQUE partial index over
active (non-archived) rows plus IntegrityError recovery in ``create_task``
make the losing writer return the winner's id instead of inserting a duplicate.
"""

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


def test_create_task_idempotency_key_is_race_safe(kanban_home):
    """Two concurrent create_task calls with one idempotency_key -> one row."""
    key = "dod:fix:v1:race-safe-check"
    barrier = threading.Barrier(2)
    results: dict = {}
    errors: dict = {}

    def worker(n: int) -> None:
        try:
            barrier.wait(timeout=10)
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
