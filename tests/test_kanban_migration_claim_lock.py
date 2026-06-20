"""Regression tests for the kanban legacy-DB migration self-heal.

Covers two defects fixed together:

* **Defect A** — ``_migrate_add_optional_columns`` ALTERed in the other
  ~18 optional columns for legacy DBs but OMITTED ``claim_lock`` /
  ``claim_expires`` (those only existed in ``SCHEMA_SQL`` for fresh DBs).
  The one-shot backfill SELECT references ``claim_lock`` *before* any step
  guaranteed it existed, so a board predating the claim_lock columns
  crashed the dispatcher tick with
  ``sqlite3.OperationalError: no such column: claim_lock``.

* **Defect B** — ``_cross_process_init_lock`` HARD-RAISED on a
  ``PermissionError`` when the ``.init.lock`` sidecar wasn't writable,
  silently blocking ALL migration on that board forever. It must now log a
  warning and degrade (proceed without the advisory lock).

Why: legacy/drifted boards must self-heal on the next ``connect()`` with
the patched code — no manual DB surgery, and an unwritable lock sidecar
must not wedge kanban on the board indefinitely.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# Minimal legacy ``tasks`` shape: the v1 NOT NULL columns plus the columns
# the backfill SELECT reads, but DELIBERATELY missing ``claim_lock`` /
# ``claim_expires`` (and the other post-v1 additive columns) to simulate a
# board created before those columns were introduced.
_LEGACY_TASKS_DDL = """
CREATE TABLE tasks (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    body            TEXT,
    assignee        TEXT,
    status          TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    started_at      INTEGER
)
"""

# ``task_runs`` must exist for the backfill branch to run; reuse the same
# shape as the live schema for the columns the backfill INSERTs into.
_TASK_RUNS_DDL = """
CREATE TABLE task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
)
"""


# ``task_events`` gets an additive ``run_id`` column in the same migration;
# it must exist for the pass to complete.
_TASK_EVENTS_DDL = """
CREATE TABLE task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    created_at INTEGER NOT NULL
)
"""


def _legacy_conn() -> sqlite3.Connection:
    """Build an in-memory connection shaped like a pre-claim_lock board.

    Why: isolates the migration function from the full schema/WAL/connect
    machinery so the assertion targets exactly the column-omission defect.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        _LEGACY_TASKS_DDL + ";" + _TASK_RUNS_DDL + ";" + _TASK_EVENTS_DDL
    )
    return conn


def test_migration_self_heals_missing_claim_lock_columns():
    """Defect A: legacy board with a running task migrates without crashing.

    Why: a board predating ``claim_lock`` must self-heal on connect.
    What: seeds a 'running' task, runs the migration, asserts no exception,
    the columns now exist, and the backfill produced a matching run row.
    Test: this test — it fails on unpatched code with OperationalError:
    no such column: claim_lock.
    """
    conn = _legacy_conn()
    now = int(time.time())
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, started_at) "
        "VALUES (?, ?, 'running', ?, ?)",
        ("t-legacy-1", "legacy running task", now, now),
    )
    conn.commit()

    # Sanity: the columns really are absent before migration.
    pre = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "claim_lock" not in pre
    assert "claim_expires" not in pre

    # (a) no exception — the crux of Defect A.
    kb._migrate_add_optional_columns(conn)

    # (b) the columns now exist.
    post = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "claim_lock" in post
    assert "claim_expires" in post

    # (c) the backfill ran: the in-flight 'running' task got a task_runs row
    #     and a current_run_id pointer.
    run = conn.execute(
        "SELECT task_id, status FROM task_runs WHERE task_id = ?",
        ("t-legacy-1",),
    ).fetchone()
    assert run is not None
    assert run["status"] == "running"
    ptr = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", ("t-legacy-1",)
    ).fetchone()
    assert ptr["current_run_id"] is not None
    conn.close()


def test_migration_is_idempotent_on_second_run():
    """Defect A guard: running the migration twice stays clean.

    Why: ``connect()``/``init_db`` may re-run the pass; it must not crash or
    double-backfill.
    What: runs the migration twice; asserts no exception and exactly one run
    row for the in-flight task.
    Test: this test — a non-idempotent ALTER or backfill would raise or
    duplicate.
    """
    conn = _legacy_conn()
    now = int(time.time())
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at, started_at) "
        "VALUES (?, ?, 'running', ?, ?)",
        ("t-legacy-2", "legacy running task", now, now),
    )
    conn.commit()

    kb._migrate_add_optional_columns(conn)
    # Real re-migration (init_db) re-runs against a freshly-opened
    # connection; commit any pending implicit DDL txn so the second pass
    # starts clean, mirroring that.
    conn.commit()
    kb._migrate_add_optional_columns(conn)

    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?",
        ("t-legacy-2",),
    ).fetchone()
    assert rows["n"] == 1
    conn.close()


def test_init_lock_degrades_when_sidecar_unwritable(tmp_path, monkeypatch, caplog):
    """Defect B: an unwritable ``.init.lock`` must not block migration.

    Why: a non-writable lock sidecar previously raised and wedged ALL
    migration on the board forever.
    What: forces the sidecar ``open`` to raise PermissionError, then runs a
    full ``connect()``; asserts it still returns a usable, fully-migrated DB
    and that a warning was logged (not raised).
    Test: this test — on unpatched code ``connect()`` propagates
    PermissionError instead of returning.
    """
    db_path = tmp_path / "kanban.db"
    lock_name = db_path.name + ".init.lock"

    real_open = Path.open

    def fake_open(self, *args, **kwargs):
        if self.name == lock_name:
            raise PermissionError(f"simulated unwritable lock: {self}")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    with caplog.at_level("WARNING"):
        # Must NOT raise even though the advisory lock can't be taken.
        conn = kb.connect(db_path=db_path)

    try:
        # The board is fully usable: the migrated schema is present.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "claim_lock" in cols
        assert "claim_expires" in cols
    finally:
        conn.close()

    # Degraded gracefully with a warning, not a raise.
    assert any(
        "init lock" in rec.message.lower() for rec in caplog.records
    ), [rec.message for rec in caplog.records]
