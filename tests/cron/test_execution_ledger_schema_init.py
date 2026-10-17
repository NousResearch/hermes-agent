"""The execution ledger creates its schema once per ledger file per process (#135487).

``cron.executions._connect`` runs once per transaction, and every adopt and ledger write opens a
fresh connection. Re-issuing the schema DDL on each of those opens put nine statements in front of
every operation; these tests pin the two properties that make skipping it safe: an uninitialized or
recreated file is always initialized, and an already initialized one is not re-issued.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_DDL_PREFIXES = ("CREATE TABLE", "CREATE INDEX", "ALTER TABLE")


def _point_ledger(monkeypatch, tmp_path):
    from cron import executions

    # Each test gets its own temporary ledger file, so its identity (device, inode) is cold for the
    # per-process schema memo with no test-only reach into the implementation.
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def _trace_ddl(monkeypatch) -> list[str]:
    """Collect every statement the ledger connection executes, so a test can assert which DDL ran."""
    seen: list[str] = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(seen.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return seen


def _ddl(statements: list[str]) -> list[str]:
    return [s for s in statements if s.lstrip().upper().startswith(_DDL_PREFIXES)]


def test_schema_ddl_is_not_reissued_for_an_initialized_ledger(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    seen = _trace_ddl(monkeypatch)

    executions.create_execution("job-1", source="builtin")
    assert _ddl(seen), "the first open must create the schema"

    seen.clear()
    executions.create_execution("job-1", source="builtin")
    assert _ddl(seen) == [], "an already initialized ledger must not re-issue its schema DDL"


def test_recreated_ledger_file_is_initialized_again(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    path = tmp_path / "cron" / "executions.db"
    executions.create_execution("job-1", source="builtin")

    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)

    seen = _trace_ddl(monkeypatch)
    record = executions.create_execution("job-1", source="builtin")

    assert _ddl(seen), "a recreated ledger file is a new file and must be initialized again"
    assert record["status"] == "claimed"


def test_pre_existing_legacy_ledger_is_migrated_on_first_open(monkeypatch, tmp_path):
    """A ledger written by an older build (no MEMO entry, older columns) still gets its ALTERs."""
    executions = _point_ledger(monkeypatch, tmp_path)
    path = tmp_path / "cron" / "executions.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(path)
    try:
        legacy.execute(
            """CREATE TABLE executions (
                 id TEXT PRIMARY KEY,
                 job_id TEXT NOT NULL,
                 source TEXT NOT NULL,
                 process_id TEXT NOT NULL,
                 pid INTEGER NOT NULL,
                 process_started_at INTEGER,
                 status TEXT NOT NULL,
                 claimed_at TEXT NOT NULL,
                 started_at TEXT,
                 finished_at TEXT,
                 error TEXT
               )"""
        )
        legacy.commit()
    finally:
        legacy.close()

    record = executions.create_execution("job-1", source="builtin")
    assert record["status"] == "claimed"

    conn = sqlite3.connect(path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(executions)")}
    finally:
        conn.close()
    assert {
        "handoff_pending",
        "handoff_started_at",
        "delivery_outcome",
        "scheduled_instant",
        "progress_at",
    } <= columns
