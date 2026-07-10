from __future__ import annotations

import json
import multiprocessing
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_safety as safety


def _hold_posix_exclusive_lock(lock_path: str, ready, release) -> None:
    import fcntl

    with open(lock_path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(5)


def _fresh_db(tmp_path: Path) -> Path:
    path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with kb.connect(path):
        pass
    return path


def test_windows_directory_durability_adapter_is_supported_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(safety.sys, "platform", "win32")
    opened: list[Path] = []

    def unexpected_open(path, _flags):
        opened.append(Path(path))
        raise AssertionError("Windows must not open a directory for fsync")

    monkeypatch.setattr(safety.os, "open", unexpected_open)
    fsync_directory = getattr(safety, "_fsync_parent_directory")
    fsync_directory(tmp_path)
    assert opened == []


def test_cache_hit_still_checks_persistent_quarantine(tmp_path):
    db_path = _fresh_db(tmp_path)
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

    marker = safety.quarantine_board(db_path, reason="integrity failed", source="test")

    with pytest.raises(safety.BoardQuarantinedError, match="integrity failed"):
        kb.connect(db_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["reason"] == "integrity failed"
    assert payload["source"] == "test"
    assert payload["timestamp"]
    assert payload["db_fingerprint"]["db"]["sha256"]


def test_corrupt_header_creates_persistent_marker(tmp_path):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"definitely not sqlite")

    with pytest.raises(safety.BoardQuarantinedError):
        kb.connect(db_path)

    marker = safety.quarantine_marker_path(db_path)
    assert marker.exists()
    assert "invalid SQLite header" in json.loads(marker.read_text())["reason"]


def test_transient_busy_does_not_create_quarantine_marker(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    monkeypatch.setattr(kb, "_resolve_busy_timeout_ms", lambda: 10)
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            kb.connect(db_path)
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert not safety.quarantine_marker_path(db_path).exists()


def test_disk_io_is_corruption_but_busy_is_not():
    assert kb._is_malformed_database_error(sqlite3.OperationalError("disk I/O error"))
    assert not kb._is_malformed_database_error(
        sqlite3.OperationalError("database is locked")
    )


def test_write_txn_disk_io_creates_persistent_quarantine(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    conn = sqlite3.connect(db_path, isolation_level=None)

    def fail_boundary(_conn, _sql):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(kb, "_execute_boundary_with_retry", fail_boundary)
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            with kb.write_txn(conn):
                pass
    finally:
        conn.close()

    marker = json.loads(safety.quarantine_marker_path(db_path).read_text())
    assert marker["source"] == "kanban_db.write_txn"


def test_write_txn_busy_does_not_create_quarantine(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    conn = sqlite3.connect(db_path, isolation_level=None)

    def fail_boundary(_conn, _sql):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb, "_execute_boundary_with_retry", fail_boundary)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with kb.write_txn(conn):
                pass
    finally:
        conn.close()

    assert not safety.quarantine_marker_path(db_path).exists()


def test_quarantine_marker_write_failure_is_explicit_and_fail_closed(
    tmp_path, monkeypatch
):
    db_path = _fresh_db(tmp_path)

    def fail_marker(*_args, **_kwargs):
        raise OSError("fsync failed")

    monkeypatch.setattr(safety, "quarantine_board", fail_marker)
    persistence_error = getattr(safety, "QuarantinePersistenceError")
    persist_quarantine = getattr(kb, "_persist_quarantine")
    with pytest.raises(persistence_error, match="failed to persist"):
        persist_quarantine(
            db_path,
            sqlite3.DatabaseError("database disk image is malformed"),
            source="test",
        )


def test_exclusive_maintenance_lock_blocks_writer(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    monkeypatch.setattr(safety, "DEFAULT_LOCK_TIMEOUT_SECONDS", 0.05)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        with safety.maintenance_lock(db_path, exclusive=True):
            with pytest.raises(safety.MaintenanceLockError, match="timed out"):
                with kb.write_txn(conn):
                    conn.execute(
                        "INSERT INTO tasks (id, title, status, created_at) "
                        "VALUES ('x', 'x', 'todo', 1)"
                    )
    finally:
        conn.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock test")
def test_real_posix_exclusive_lock_blocks_shared_maintenance_lock(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Event()
    release = ctx.Event()
    process = ctx.Process(
        target=_hold_posix_exclusive_lock,
        args=(str(safety.maintenance_lock_path(db_path)), ready, release),
    )
    process.start()
    try:
        assert ready.wait(5)
        monkeypatch.setattr(safety, "DEFAULT_LOCK_TIMEOUT_SECONDS", 0.05)
        with pytest.raises(safety.MaintenanceLockError, match="timed out"):
            with safety.maintenance_lock(db_path, exclusive=False):
                pass
    finally:
        release.set()
        process.join(5)
    assert process.exitcode == 0


def test_write_txn_lock_open_failure_is_fail_closed(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    conn = sqlite3.connect(db_path, isolation_level=None)

    def fail_open(_path):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(safety, "_open_lock_file", fail_open)
    try:
        with pytest.raises(safety.MaintenanceLockError, match="open"):
            with kb.write_txn(conn):
                pass
    finally:
        conn.close()


def test_write_txn_lock_acquire_failure_is_fail_closed(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    conn = sqlite3.connect(db_path, isolation_level=None)

    def fail_acquire(_handle, *, exclusive):
        raise OSError("flock unavailable")

    monkeypatch.setattr(safety, "_try_lock", fail_acquire)
    try:
        with pytest.raises(safety.MaintenanceLockError, match="acquire"):
            with kb.write_txn(conn):
                pass
    finally:
        conn.close()


def test_unlock_failure_does_not_mask_body_corruption(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"

    def fail_unlock(_handle):
        raise OSError("unlock failed")

    monkeypatch.setattr(safety, "_unlock", fail_unlock)
    with pytest.raises(sqlite3.DatabaseError, match="malformed") as exc_info:
        with safety.maintenance_lock(db_path, exclusive=False):
            raise sqlite3.DatabaseError("database disk image is malformed")

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("failed to release maintenance lock" in note for note in notes)


def test_unlock_failure_without_body_error_is_fail_closed(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"

    def fail_unlock(_handle):
        raise OSError("unlock failed")

    monkeypatch.setattr(safety, "_unlock", fail_unlock)
    with pytest.raises(safety.MaintenanceLockError, match="release"):
        with safety.maintenance_lock(db_path, exclusive=False):
            pass


def test_stale_service_and_board_generations_are_fenced(tmp_path):
    db_path = tmp_path / "kanban.db"
    initial = safety.read_generations(db_path)
    current_service = safety.bump_service_generation(db_path)
    current_board = safety.bump_board_generation(db_path)

    with pytest.raises(safety.GenerationFencedError, match="service_generation"):
        safety.validate_generations(
            db_path, expected_service_generation=initial.service_generation
        )
    with pytest.raises(safety.GenerationFencedError, match="board_generation"):
        safety.validate_generations(
            db_path, expected_board_generation=initial.board_generation
        )
    assert current_service == initial.service_generation + 1
    assert current_board == initial.board_generation + 1


@pytest.mark.parametrize("invalid", [True, False, 1.5])
def test_generation_metadata_rejects_non_integer_json_values(tmp_path, invalid):
    db_path = tmp_path / "kanban.db"
    safety.generations_path(db_path).write_text(
        json.dumps({"service_generation": invalid, "board_generation": 0}),
        encoding="utf-8",
    )

    with pytest.raises(safety.KanbanSafetyError, match="malformed"):
        safety.read_generations(db_path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"service_generation": 0},
        {"board_generation": 0},
    ],
)
def test_existing_generation_metadata_requires_both_fields(tmp_path, payload):
    db_path = tmp_path / "kanban.db"
    safety.generations_path(db_path).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(safety.KanbanSafetyError, match="malformed"):
        safety.read_generations(db_path)


@pytest.mark.parametrize("invalid", [True, False, 1.5])
def test_validate_generations_rejects_non_integer_expected_values(tmp_path, invalid):
    db_path = tmp_path / "kanban.db"

    with pytest.raises(safety.KanbanSafetyError, match="malformed"):
        safety.validate_generations(
            db_path, expected_service_generation=invalid
        )


def test_quarantine_snapshot_blocks_generation_bump(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path)
    fingerprint_started = threading.Event()
    release_fingerprint = threading.Event()
    errors: list[BaseException] = []
    real_fingerprint = safety.db_fingerprint

    def slow_fingerprint(path):
        fingerprint_started.set()
        assert release_fingerprint.wait(5)
        return real_fingerprint(path)

    def create_marker():
        try:
            safety.quarantine_board(db_path, reason="race", source="test")
        except BaseException as exc:  # surfaced below in the main test thread
            errors.append(exc)

    monkeypatch.setattr(safety, "db_fingerprint", slow_fingerprint)
    thread = threading.Thread(target=create_marker)
    thread.start()
    assert fingerprint_started.wait(5)
    monkeypatch.setattr(safety, "DEFAULT_LOCK_TIMEOUT_SECONDS", 0.05)
    try:
        with pytest.raises(safety.MaintenanceLockError, match="timed out"):
            safety.bump_board_generation(db_path)
    finally:
        release_fingerprint.set()
        thread.join(5)

    assert not thread.is_alive()
    assert errors == []
    marker = safety.active_quarantine(db_path)
    assert marker is not None
    assert marker["board_generation"] == safety.read_generations(db_path).board_generation


def test_marker_must_be_explicitly_cleared_before_same_generation_recovers(tmp_path):
    db_path = _fresh_db(tmp_path)
    marker = safety.quarantine_board(db_path, reason="manual hold", source="test")
    fingerprint = json.loads(marker.read_text())["db_fingerprint"]

    with pytest.raises(safety.BoardQuarantinedError):
        kb.connect(db_path)
    safety.clear_quarantine(db_path, expected_fingerprint=fingerprint)
    with kb.connect(db_path) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_corrupt_database_cannot_be_cleared_with_marker_fingerprint(tmp_path):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"not sqlite")
    marker = safety.quarantine_board(db_path, reason="corrupt", source="test")
    fingerprint = json.loads(marker.read_text())["db_fingerprint"]

    with pytest.raises(safety.KanbanSafetyError, match="health verification"):
        safety.clear_quarantine(db_path, expected_fingerprint=fingerprint)

    assert marker.exists()


def test_clear_quarantine_requires_current_schema_indexes(tmp_path):
    db_path = _fresh_db(tmp_path)
    marker = safety.quarantine_board(db_path, reason="maintenance", source="test")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_notify_task")
    fingerprint = safety.db_fingerprint(db_path)

    with pytest.raises(safety.KanbanSafetyError, match="canonical schema"):
        safety.clear_quarantine(db_path, expected_fingerprint=fingerprint)
    assert marker.exists()


def test_clear_rejects_missing_additive_index(tmp_path):
    db_path = _fresh_db(tmp_path)
    marker = safety.quarantine_board(db_path, reason="maintenance", source="test")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_events_run")
    fingerprint = safety.db_fingerprint(db_path)

    with pytest.raises(safety.KanbanSafetyError, match="idx_events_run"):
        safety.clear_quarantine(db_path, expected_fingerprint=fingerprint)
    assert marker.exists()


def test_clear_rejects_wrong_index_definition(tmp_path):
    db_path = _fresh_db(tmp_path)
    marker = safety.quarantine_board(db_path, reason="maintenance", source="test")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_events_run")
        conn.execute("CREATE INDEX idx_events_run ON task_events(task_id, id)")
    fingerprint = safety.db_fingerprint(db_path)

    with pytest.raises(safety.KanbanSafetyError, match="idx_events_run"):
        safety.clear_quarantine(db_path, expected_fingerprint=fingerprint)
    assert marker.exists()


def test_canonical_validator_rejects_wrong_column_shape():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE tasks(id INTEGER PRIMARY KEY)")
        validate_schema = getattr(safety, "_validate_required_schema")
        with pytest.raises(safety.KanbanSafetyError, match="tasks.id"):
            validate_schema(conn)
    finally:
        conn.close()


def test_canonical_schema_contract_matches_fresh_db(tmp_path):
    db_path = _fresh_db(tmp_path)
    required_tables = getattr(safety, "_REQUIRED_TABLE_SHAPES")
    required_indexes = getattr(safety, "_REQUIRED_INDEX_SHAPES")
    with sqlite3.connect(db_path) as conn:
        actual_indexes = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type='index' AND sql IS NOT NULL"
            )
        }
        assert actual_indexes == set(required_indexes)
        for table, expected_columns in required_tables.items():
            actual_columns = {
                str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')
            }
            assert actual_columns == set(expected_columns)


def test_db_fingerprint_includes_wal_and_shm_contents(tmp_path):
    db_path = tmp_path / "kanban.db"
    db_path.write_bytes(b"db")
    wal_path = tmp_path / "kanban.db-wal"
    shm_path = tmp_path / "kanban.db-shm"
    wal_path.write_bytes(b"wal-one")
    shm_path.write_bytes(b"shm-one")

    first = safety.db_fingerprint(db_path)
    wal_path.write_bytes(b"wal-two")
    second = safety.db_fingerprint(db_path)

    assert first["db"]["sha256"] == second["db"]["sha256"]
    assert first["wal"]["sha256"] != second["wal"]["sha256"]
    assert first["shm"]["sha256"] == second["shm"]["sha256"]


def test_board_generation_bump_fences_old_marker(tmp_path):
    db_path = _fresh_db(tmp_path)
    safety.quarantine_board(db_path, reason="rebuild required", source="test")
    with pytest.raises(safety.BoardQuarantinedError):
        kb.connect(db_path)

    safety.bump_board_generation(db_path)
    with kb.connect(db_path) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
