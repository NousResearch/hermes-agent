"""Safety tests for the real SQLite state migration source adapter."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from state_store.sqlite.migration_adapter import SQLiteMigrationSourceAdapter


def _create_source(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT REFERENCES sessions(id),
                handoff_state TEXT
            );
            CREATE TABLE compression_locks (
                session_id TEXT PRIMARY KEY,
                expires_at REAL NOT NULL
            );
            CREATE TABLE async_delegations (
                delegation_id TEXT PRIMARY KEY,
                completed_at REAL,
                delivery_state TEXT
            );
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT);
            """
        )
        connection.executemany(
            "INSERT INTO sessions (id, parent_session_id, handoff_state) VALUES (?, ?, ?)",
            [
                ("z-parent", None, "pending"),
                ("a-child", "z-parent", None),
            ],
        )
        connection.executemany(
            "INSERT INTO messages (id, session_id) VALUES (?, ?)",
            [(1, "a-child"), (2, "z-parent"), (3, "z-parent")],
        )
        connection.execute(
            "INSERT INTO compression_locks (session_id, expires_at) VALUES (?, ?)",
            ("a-child", 200.0),
        )
        connection.execute(
            """
            INSERT INTO async_delegations (delegation_id, completed_at, delivery_state)
            VALUES (?, ?, ?)
            """,
            ("delegation-1", None, "pending"),
        )
        connection.commit()
    finally:
        connection.close()


def test_writer_fence_blocks_another_sqlite_writer_without_a_runtime_hook(tmp_path: Path):
    path = tmp_path / "state.db"
    _create_source(path)

    adapter = SQLiteMigrationSourceAdapter(path, fence_timeout_s=0.2)

    with adapter.writer_fence("run-1"):
        contender = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sqlite3, sys; "
                    "connection = sqlite3.connect(sys.argv[1], timeout=0.1, isolation_level=None); "
                    "connection.execute('PRAGMA busy_timeout = 100'); "
                    "\ntry:\n"
                    "    connection.execute('BEGIN IMMEDIATE')\n"
                    "except sqlite3.OperationalError:\n"
                    "    raise SystemExit(0)\n"
                    "raise SystemExit(1)\n"
                ),
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    assert contender.returncode == 0, contender.stderr


def test_backup_snapshot_uses_backup_api_and_keyset_batches_without_wal_copy(
    tmp_path: Path,
):
    path = tmp_path / "state.db"
    _create_source(path)
    # A WAL sidecar must never be copied as a migration snapshot input.
    wal = tmp_path / "state.db-wal"
    wal.write_bytes(b"not a SQLite WAL")
    fenced: list[str] = []

    @contextmanager
    def writer_fence(run_id: str):
        fenced.append(run_id)
        yield

    adapter = SQLiteMigrationSourceAdapter(path, writer_fence_hook=writer_fence)
    with pytest.raises(RuntimeError, match="writer fence"):
        adapter.snapshot_via_sqlite_backup("without-fence")
    with adapter.writer_fence("run-2"):
        snapshot = adapter.snapshot_via_sqlite_backup("run-2")
        try:
            assert snapshot._path != path  # noqa: SLF001 - verifies temporary backup boundary
            assert not snapshot._path.with_name(snapshot._path.name + "-wal").exists()  # noqa: SLF001
            first = snapshot.fetchmany_keyset("sessions", ("id",), None, 1)
            second = snapshot.fetchmany_keyset("sessions", ("id",), ("a-child",), 1)
            assert [row["id"] for row in first] == ["a-child"]
            assert [row["id"] for row in second] == ["z-parent"]
        finally:
            backup_path = snapshot._path  # noqa: SLF001 - cleanup contract is public behavior
            snapshot.close()
    assert fenced == ["run-2"]
    assert not backup_path.exists()


def test_preflight_checks_integrity_foreign_keys_and_active_records(tmp_path: Path):
    path = tmp_path / "state.db"
    _create_source(path)
    adapter = SQLiteMigrationSourceAdapter(path, writer_fence_hook=lambda _: _null_fence())

    assert adapter.sqlite_integrity_errors() == []
    assert adapter.sqlite_foreign_key_violations() == []
    assert adapter.active_compression_leases(100.0) == ["a-child"]
    assert adapter.active_delegations() == ["delegation-1"]
    assert adapter.active_handoffs() == ["z-parent"]

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "INSERT INTO sessions (id, parent_session_id, handoff_state) VALUES (?, ?, ?)",
            ("orphan", "missing-parent", None),
        )
        connection.commit()
    finally:
        connection.close()
    assert adapter.sqlite_foreign_key_violations() == [
        "sessions rowid=3 parent=sessions fk=0"
    ]


@contextmanager
def _null_fence():
    yield
