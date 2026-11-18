"""Declarative schema drift on ``MaintenanceAdmin``: a read-only handle on a lagging store refuses the explicit repair and the browse reconcile, and leaves the file as it was."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from persistence import PermanentStoreError, ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.admin import SQLiteStoreAdmin
from tests.fixtures.storage_seed import seed_messages, seed_session

_ADDED_TABLES = ("runtime_epoch", "session_admissions", "worker_executions", "worker_receipts")
_ADDED_COLUMNS = (("sessions", "runtime_revision"), ("sessions", "runtime_generation"),
                  ("async_delegations", "owner_execution_id"))


@pytest.fixture(autouse=True)
def _volume_headroom(monkeypatch):
    """The backup and snapshot guards reserve 2% of the volume, and the holder scan sees every process on the host."""
    from persistence.sqlite._state import repair
    monkeypatch.setattr(repair, "_REPAIR_BACKUP_FREE_FRACTION", 0.0)
    monkeypatch.setattr(repair, "_live_writer_holds_db", lambda _path: False)


def _store_lagging(path: Path) -> Path:
    """A store with one session's history written through the backend's store, its four declared tables and three declared columns removed again."""
    scope = ProfileScope(profile_dir=path.parent, name="default")
    backend = SQLiteBackend({scope.hermes_home_key: path})
    try:
        store = backend.open(scope)
        seed_session(store, "kept", "cli")
        seed_messages(store, "kept", {"role": "user", "content": "history survives"})
    finally:
        backend.close()
    return _drop_added_schema(path)


def _drop_added_schema(path: Path) -> Path:
    """Remove the four declared tables and three declared columns again."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TRIGGER IF EXISTS sessions_runtime_ledger_delete_guard")
        for table in reversed(_ADDED_TABLES):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        for table, column in _ADDED_COLUMNS:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.commit()
    return path


def _homes(tmp_path: Path, build) -> tuple[Path, Path]:
    """The same store in two profile homes: the legacy side's and the admin's."""
    seed = build(tmp_path / "seed.db")
    paths = []
    for name in ("legacy", "admin"):
        home = tmp_path / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
        with closing(sqlite3.connect(seed)) as src, closing(sqlite3.connect(home / "state.db")) as dst:
            src.backup(dst)
        paths.append(home / "state.db")
    return paths[0], paths[1]


# ── read-only refusals ─────────────────────────────────────────────────────


@pytest.mark.parametrize("operation", ["repair_declarative_schema", "reconcile_drift_for_browse"])
def test_read_only_handle_refuses_schema_writes(tmp_path, operation):
    _legacy_path, admin_path = _homes(tmp_path, _store_lagging)
    before = admin_path.read_bytes()
    with pytest.raises(PermanentStoreError):
        getattr(SQLiteStoreAdmin(admin_path, read_only=True), operation)()
    assert admin_path.read_bytes() == before
