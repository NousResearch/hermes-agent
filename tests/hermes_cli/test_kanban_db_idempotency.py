"""Storage contracts for Kanban idempotency and legacy migration."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_task_idempotent_reports_replay(kanban_home):
    with kbc.connect() as conn:
        first_id, first_created = kb.create_task_idempotent(
            conn, title="original", idempotency_key="replay-key"
        )
        replay_id, replay_created = kb.create_task_idempotent(
            conn, title="retry of original", idempotency_key="replay-key"
        )
    assert first_created is True
    assert replay_created is False
    assert replay_id == first_id



def test_empty_idempotency_key_is_no_key(kanban_home):
    """``""`` never deduped, so it must not occupy the unique index either."""
    with kbc.connect() as conn:
        first = kb.create_task(conn, title="one", idempotency_key="")
        second = kb.create_task(conn, title="two", idempotency_key="")
    assert first != second
    with kbc.connect() as conn:
        assert all(row[0] is None for row in conn.execute(
            "SELECT idempotency_key FROM tasks WHERE id IN (?, ?)", (first, second)))



def test_unique_idempotency_migration_holds_out_concurrent_writers(kanban_home):
    """Dedupe + index build are one transaction: a writer still on the pre-index code cannot
    slip a duplicate key in between and fail the build."""
    with kbc.connect() as conn:
        for task_id in ("t_a", "t_b"):
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
                "VALUES (?, 'legacy', 'ready', 1, 'scratch')", (task_id,))
        conn.execute("DROP INDEX idx_tasks_idempotency_unique")
        other = sqlite3.connect(kb.kanban_db_path(), isolation_level=None, timeout=0)
        refused: list[Exception] = []

        def _old_writer(statement: str) -> None:
            if statement.startswith("CREATE UNIQUE INDEX"):
                try:
                    other.execute("UPDATE tasks SET idempotency_key = 'dup' WHERE id IN ('t_a', 't_b')")
                except sqlite3.OperationalError as exc:
                    refused.append(exc)

        conn.set_trace_callback(_old_writer)
        try:
            kbc._migrate_unique_idempotency_index(conn)
        finally:
            conn.set_trace_callback(None)
            other.close()
        assert refused
        assert "idx_tasks_idempotency_unique" in {
            row["name"] for row in conn.execute("PRAGMA index_list(tasks)")}



def test_create_task_idempotent_resolves_lost_unique_index_race(
    kanban_home, monkeypatch
):
    """A creator that loses the UNIQUE-index race returns the winner's id.

    The concurrent winner is injected through the ``_new_task_id`` seam,
    which runs exactly between the loser's fast-path key lookup and its
    INSERT — so the loser's INSERT hits the partial UNIQUE index
    deterministically and must recover to ``(winner_id, False)`` instead of
    raising IntegrityError.
    """
    real_new_id = kb._new_task_id
    state = {"injected": False}

    def winner_commits_then_new_id():
        if not state["injected"]:
            state["injected"] = True
            other = kbc.connect()
            try:
                with kb.write_txn(other):
                    other.execute(
                        "INSERT INTO tasks "
                        "(id, title, status, created_at, workspace_kind, idempotency_key) "
                        "VALUES ('t_winner', 'winner', 'ready', 1, 'scratch', 'race-key')"
                    )
            finally:
                other.close()
        return real_new_id()

    monkeypatch.setattr(kb, "_new_task_id", winner_commits_then_new_id)

    with kbc.connect() as conn:
        task_id, created = kb.create_task_idempotent(
            conn, title="loser", idempotency_key="race-key"
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = 'race-key'"
        ).fetchone()[0]

    assert (task_id, created) == ("t_winner", False)
    assert count == 1



def test_concurrent_creates_with_one_key_yield_one_task(kanban_home, monkeypatch):
    """Stress-contract regression (idempotency_key_race in tests/stress):
    concurrent create_task calls with one idempotency key must all return
    the same id and persist a single row."""
    barrier = threading.Barrier(4)
    real_new_id = kb._new_task_id

    def synchronized_id():
        barrier.wait(timeout=10)
        return real_new_id()

    monkeypatch.setattr(kb, "_new_task_id", synchronized_id)
    results: list[str] = []
    errors: list[BaseException] = []

    def worker():
        try:
            conn = kbc.connect()
            try:
                results.append(
                    kb.create_task(conn, title="racer", idempotency_key="stampede")
                )
            finally:
                conn.close()
        except BaseException as exc:  # surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert len(results) == 4
    assert len(set(results)) == 1
    with kbc.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = 'stampede'"
        ).fetchone()[0]
    assert count == 1



def test_migration_preserves_cards_and_only_newest_live_key(kanban_home):
    with kbc.connect() as conn:
        conn.execute("DROP INDEX idx_tasks_idempotency_unique")
        conn.execute("CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)")
        for task_id, status, created_at in (("t_old", "ready", 1), ("t_new", "ready", 2),
                                             ("t_archived", "archived", 3)):
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at, workspace_kind, idempotency_key) "
                "VALUES (?, 'legacy', ?, ?, 'scratch', 'duplicate')", (task_id, status, created_at))
        kbc._migrate_unique_idempotency_index(conn)
        rows = dict(conn.execute("SELECT id, idempotency_key FROM tasks"))
        assert rows == {"t_old": None, "t_new": "duplicate", "t_archived": "duplicate"}
        assert kb.create_task(conn, title="replay", idempotency_key="duplicate") == "t_new"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE tasks SET idempotency_key = 'duplicate' WHERE id = 't_old'")
