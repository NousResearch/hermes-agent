"""SQLite persistence with preflight validation and fail-closed recovery."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from plugins.agentops.control.audit import audit_entry_hash, validate_audit_event
from plugins.agentops.control.config import AgentOpsConfig, path_is_within_state
from plugins.agentops.control.events import canonical_json
from plugins.agentops.control.models import AppendResult, AuditEvent, EventEnvelope, StoreInspection


SCHEMA_VERSION = 1
_REQUIRED_TABLES = {"schema_migrations", "events", "audit_events", "metadata"}


class StoreMigrationError(RuntimeError):
    """A database cannot safely be opened by this Phase 1 store."""


class StoreIntegrityError(RuntimeError):
    """Stored data violates an append-only invariant."""


class StoreRestoreError(StoreMigrationError):
    """A restore was rejected or was rolled back to its original database."""


@dataclass
class _RestoreState:
    closed: bool = False
    replaced: bool = False


class AgentOpsStore:
    def __init__(self, config: AgentOpsConfig, connection: sqlite3.Connection):
        self.config = config
        self.path = config.sqlite_path
        self._connection = connection
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _transaction(self):
        return _Transaction(self)

    def journal_mode(self) -> str:
        with self._lock:
            return str(self._connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def schema_version(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT version FROM schema_migrations WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise StoreMigrationError("schema version unavailable")
        return int(row[0])

    def append_event(self, event: EventEnvelope) -> AppendResult:
        event_json = canonical_json(event.to_dict())
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO events(event_id, event_hash, event_json, occurred_at, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.content_hash,
                    event_json,
                    event.occurred_at.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                existing = cursor.execute("SELECT event_hash FROM events WHERE event_id = ?", (event.event_id,)).fetchone()
                if existing is None or str(existing[0]) != event.content_hash:
                    raise StoreIntegrityError("event identity conflict")
            if inserted:
                self._append_audit_locked(
                    cursor,
                    AuditEvent.create(
                        actor_type="system",
                        actor_id="agentopsd",
                        action="event.append",
                        object_type="event",
                        object_id=event.event_id,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        metadata={"event_hash": event.content_hash},
                        after_hash=event.content_hash,
                    ),
                )
        return AppendResult(event_id=event.event_id, inserted=inserted, content_hash=event.content_hash)

    def _append_audit_locked(self, cursor: sqlite3.Cursor, event: AuditEvent) -> int:
        validate_audit_event(event)
        if not _verify_chain_read_only(self._connection):
            raise StoreIntegrityError("audit chain invalid")
        head_sequence, head_hash = _read_audit_head(cursor)
        row = cursor.execute("SELECT COUNT(*), MAX(sequence), MAX(entry_hash) FROM audit_events").fetchone()
        count = int(row[0])
        if count != head_sequence:
            raise StoreIntegrityError("audit metadata mismatch")
        if count == 0:
            if head_hash:
                raise StoreIntegrityError("audit metadata mismatch")
            previous_hash: str | None = None
        else:
            tail = cursor.execute("SELECT sequence, entry_hash FROM audit_events ORDER BY sequence DESC LIMIT 1").fetchone()
            if tail is None or int(tail[0]) != head_sequence or str(tail[1]) != head_hash:
                raise StoreIntegrityError("audit metadata mismatch")
            previous_hash = head_hash
        next_sequence = head_sequence + 1
        payload = event.to_dict(previous_hash=previous_hash)
        entry_hash = audit_entry_hash(sequence=next_sequence, payload=payload)
        cursor.execute(
            "INSERT INTO audit_events(sequence, event_json, previous_hash, entry_hash) VALUES (?, ?, ?, ?)",
            (next_sequence, canonical_json(payload), previous_hash, entry_hash),
        )
        _write_audit_head(cursor, next_sequence, entry_hash)
        return next_sequence

    def append_audit(self, event: AuditEvent) -> int:
        with self._transaction() as cursor:
            return self._append_audit_locked(cursor, event)

    def verify_audit_chain(self) -> bool:
        with self._lock:
            return _verify_chain_read_only(self._connection)

    def event_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def audit_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0])

    def backup_to(self, destination: Path | None = None) -> Path:
        if destination is None:
            destination = self.config.backup_dir / f"backup-{uuid.uuid4().hex}.db"
        destination = Path(destination)
        self._validate_backup_destination(destination)
        with self._lock:
            return self._backup_to_locked(destination)

    def _validate_backup_destination(self, destination: Path) -> None:
        if not path_is_within_state(self.config, destination) or not _is_within(destination, self.config.backup_dir):
            raise StoreMigrationError("backup destination rejected")
        if destination.exists() or destination.is_symlink():
            raise StoreMigrationError("backup destination rejected")
        self.config.backup_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        status = self.config.backup_dir.lstat()
        if (
            status.st_uid != os.getuid()
            or not stat.S_ISDIR(status.st_mode)
            or stat.S_ISLNK(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o700
        ):
            raise StoreMigrationError("backup directory rejected")

    def _backup_to_locked(self, destination: Path) -> Path:
        with sqlite3.connect(destination) as backup_connection:
            self._connection.backup(backup_connection)
        _fsync_file(destination)
        _fsync_directory(destination.parent)
        return destination

    def _open_existing_writable(self) -> sqlite3.Connection:
        _preflight_database(self.path)
        return _connect_write(self.path)

    def restore_from(self, source: Path) -> None:
        """Validate a controlled backup before atomically replacing the live store.

        A pre-restore snapshot is kept even when the replacement must be rolled
        back, so an operator can investigate the failed candidate safely.
        """
        source = Path(source)
        if (
            not source.exists()
            or source.is_symlink()
            or not source.is_file()
            or not _is_within(source.resolve(strict=False), self.config.backup_dir)
        ):
            raise StoreRestoreError("backup unavailable")
        try:
            validation_copy = _copy_read_only_backup(source, self.config.backup_dir, "validate")
        except (OSError, sqlite3.Error, StoreMigrationError) as exc:
            raise StoreRestoreError("backup rejected") from exc
        try:
            _preflight_database(validation_copy)
        except (StoreMigrationError, StoreIntegrityError):
            validation_copy.unlink(missing_ok=True)
            raise StoreRestoreError("backup rejected")

        replacement = self.config.backup_dir / f"replace-{uuid.uuid4().hex}.db"
        snapshot: Path | None = None
        try:
            with self._lock:
                state = _RestoreState()
                try:
                    snapshot = self.config.backup_dir / f"pre-restore-{uuid.uuid4().hex}.db"
                    self._validate_backup_destination(snapshot)
                    self._backup_to_locked(snapshot)
                    _copy_database(validation_copy, replacement)
                    _preflight_database(replacement)
                    self._connection.close()
                    state.closed = True
                    os.replace(replacement, self.path)
                    state.replaced = True
                    _remove_sidecars(self.path)
                    _fsync_directory(self.path.parent)
                    self._connection = self._open_existing_writable()
                    state.closed = False
                except Exception as exc:
                    try:
                        self._recover_restore_failure_locked(snapshot, state)
                    except Exception as recovery_exc:
                        raise StoreRestoreError("restore recovery failed") from recovery_exc
                    detail = "restore rolled back" if state.replaced else "restore rejected before replacement"
                    raise StoreRestoreError(detail) from exc
        except StoreRestoreError:
            raise
        except (OSError, sqlite3.Error, StoreMigrationError) as exc:
            raise StoreRestoreError("restore failed") from exc
        finally:
            validation_copy.unlink(missing_ok=True)
            replacement.unlink(missing_ok=True)

    def _recover_restore_failure_locked(self, snapshot: Path | None, state: _RestoreState) -> None:
        """Leave the caller with a verified usable handle after a failed restore."""
        if not state.closed:
            if not self.verify_audit_chain():
                raise StoreRestoreError("existing store became invalid")
            return
        if state.replaced:
            if snapshot is None:
                raise StoreRestoreError("restore snapshot unavailable")
            rollback = self.config.backup_dir / f"rollback-{uuid.uuid4().hex}.db"
            try:
                _copy_database(snapshot, rollback)
            except (OSError, sqlite3.Error):
                _copy_database_emergency(snapshot, rollback)
            try:
                _remove_sidecars(self.path)
            except OSError:
                _remove_sidecars_emergency(self.path)
            try:
                os.replace(rollback, self.path)
            except OSError:
                _atomic_replace_emergency(rollback, self.path)
            try:
                _fsync_directory(self.path.parent)
            except OSError:
                _fsync_directory_emergency(self.path.parent)
        self._connection = self._reopen_after_recovery_locked()
        state.closed = False
        if not self.verify_audit_chain():
            raise StoreRestoreError("recovered store audit invalid")

    def _reopen_after_recovery_locked(self) -> sqlite3.Connection:
        try:
            return self._open_existing_writable()
        except Exception:
            return _emergency_open_existing_writable(self.path)


class _Transaction:
    def __init__(self, store: AgentOpsStore):
        self.store = store
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None

    def __enter__(self) -> sqlite3.Cursor:
        self.store._lock.acquire()
        self.connection = self.store._connection
        self.cursor = self.connection.cursor()
        self.cursor.execute("BEGIN IMMEDIATE")
        return self.cursor

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None:
                assert self.connection is not None
                self.connection.commit()
            else:
                assert self.connection is not None
                self.connection.rollback()
        finally:
            if self.cursor is not None:
                self.cursor.close()
            self.store._lock.release()


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _connect_write(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5, check_same_thread=False)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _emergency_open_existing_writable(path: Path) -> sqlite3.Connection:
    """One recovery retry independent of the ordinary opener seam used in tests."""
    _preflight_database(path)
    connection = sqlite3.connect(path, timeout=5, check_same_thread=False)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        _preflight_database(path)
        return connection
    except Exception:
        connection.close()
        raise


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _migration_v1(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE schema_migrations(singleton INTEGER PRIMARY KEY CHECK(singleton = 1), version INTEGER NOT NULL)"
    )
    connection.execute("INSERT INTO schema_migrations(singleton, version) VALUES (1, 1)")
    connection.execute(
        "CREATE TABLE events(event_id TEXT PRIMARY KEY, event_hash TEXT NOT NULL, event_json TEXT NOT NULL, occurred_at TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE audit_events(sequence INTEGER PRIMARY KEY, event_json TEXT NOT NULL, previous_hash TEXT, entry_hash TEXT NOT NULL)"
    )
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        (("audit_head_sequence", "0"), ("audit_head_hash", "")),
    )


_MIGRATIONS: dict[int, object] = {1: _migration_v1}


def _run_migrations(connection: sqlite3.Connection, current_version: int) -> None:
    if current_version < 0 or current_version > SCHEMA_VERSION:
        raise StoreMigrationError("unsupported schema version")
    for target in range(current_version + 1, SCHEMA_VERSION + 1):
        migration = _MIGRATIONS.get(target)
        if migration is None:
            raise StoreMigrationError("migration missing")
        migration(connection)
    if current_version != SCHEMA_VERSION:
        row = connection.execute("SELECT version FROM schema_migrations WHERE singleton = 1").fetchone()
        if row is None or int(row[0]) != SCHEMA_VERSION:
            raise StoreMigrationError("migration version mismatch")


def _validate_schema(connection: sqlite3.Connection) -> int:
    if _table_names(connection) != _REQUIRED_TABLES:
        raise StoreMigrationError("unrecognized existing database")
    columns = connection.execute("PRAGMA table_info(schema_migrations)").fetchall()
    names = {str(row[1]) for row in columns}
    if names != {"singleton", "version"}:
        raise StoreMigrationError("unsupported migration schema")
    rows = connection.execute("SELECT singleton, version FROM schema_migrations").fetchall()
    if len(rows) != 1 or int(rows[0][0]) != 1:
        raise StoreMigrationError("migration singleton invalid")
    version = int(rows[0][1])
    if version != SCHEMA_VERSION:
        raise StoreMigrationError("unsupported schema version")
    metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    if set(metadata) != {"audit_head_sequence", "audit_head_hash"}:
        raise StoreMigrationError("audit metadata unavailable")
    return version


def _preflight_database(path: Path) -> int:
    try:
        with _connect_read_only(path) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]).lower() != "ok":
                raise StoreMigrationError("sqlite integrity check failed")
            version = _validate_schema(connection)
            if not _verify_chain_read_only(connection):
                raise StoreIntegrityError("audit chain invalid")
            return version
    except (sqlite3.Error, OSError) as exc:
        raise StoreMigrationError("store preflight failed") from exc


def open_store(config: AgentOpsConfig) -> AgentOpsStore:
    """Open only an AgentOps-owned database; existing data is preflighted first."""
    if not config.state_dir_safe or not path_is_within_state(config, config.sqlite_path):
        raise StoreMigrationError("store path rejected")
    path = config.sqlite_path
    if path.exists():
        try:
            _preflight_database(path)
        except (StoreMigrationError, StoreIntegrityError):
            raise
    else:
        try:
            with sqlite3.connect(path) as bootstrap:
                _run_migrations(bootstrap, 0)
        except (sqlite3.Error, OSError, StoreMigrationError) as exc:
            raise StoreMigrationError("store bootstrap failed") from exc
    try:
        connection = _connect_write(path)
        _preflight_database(path)
        return AgentOpsStore(config, connection)
    except (sqlite3.Error, OSError, StoreMigrationError, StoreIntegrityError) as exc:
        try:
            connection.close()  # type: ignore[name-defined]
        except (UnboundLocalError, sqlite3.Error):
            pass
        raise StoreMigrationError("store open failed") from exc


def inspect_store(path: Path) -> StoreInspection:
    """Inspect through SQLite read-only mode and never initialize WAL."""
    path = Path(path)
    if not path.exists():
        return StoreInspection(False, None, None, None, None, "store_missing")
    try:
        with _connect_read_only(path) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            integrity_ok = integrity is not None and str(integrity[0]).lower() == "ok"
            version = _validate_schema(connection) if integrity_ok else None
            events = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]) if version else None
            chain = _verify_chain_read_only(connection) if version else False
        return StoreInspection(True, version, chain, events, integrity_ok, None if chain else "audit_chain_invalid")
    except (sqlite3.Error, OSError, StoreMigrationError, StoreIntegrityError):
        return StoreInspection(True, None, False, None, False, "store_invalid")


def _read_audit_head(cursor: sqlite3.Cursor) -> tuple[int, str]:
    rows = dict(cursor.execute("SELECT key, value FROM metadata WHERE key IN ('audit_head_sequence', 'audit_head_hash')"))
    try:
        return int(rows["audit_head_sequence"]), str(rows["audit_head_hash"])
    except (KeyError, ValueError) as exc:
        raise StoreIntegrityError("audit metadata unavailable") from exc


def _write_audit_head(cursor: sqlite3.Cursor, sequence: int, entry_hash: str) -> None:
    cursor.execute("UPDATE metadata SET value = ? WHERE key = 'audit_head_sequence'", (str(sequence),))
    cursor.execute("UPDATE metadata SET value = ? WHERE key = 'audit_head_hash'", (entry_hash,))
    if cursor.rowcount != 1:
        raise StoreIntegrityError("audit metadata unavailable")


def _verify_chain_read_only(connection: sqlite3.Connection) -> bool:
    try:
        rows = connection.execute(
            "SELECT sequence, event_json, previous_hash, entry_hash FROM audit_events ORDER BY sequence"
        ).fetchall()
        expected_sequence = 1
        previous_hash: str | None = None
        for sequence, event_json, recorded_previous, entry_hash in rows:
            if int(sequence) != expected_sequence or recorded_previous != previous_hash:
                return False
            payload = json.loads(event_json)
            if audit_entry_hash(sequence=int(sequence), payload=payload) != entry_hash:
                return False
            previous_hash = str(entry_hash)
            expected_sequence += 1
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
        if set(metadata) != {"audit_head_sequence", "audit_head_hash"}:
            return False
        return int(metadata["audit_head_sequence"]) == len(rows) and str(metadata["audit_head_hash"]) == (previous_hash or "")
    except (sqlite3.Error, ValueError, TypeError, json.JSONDecodeError):
        return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_emergency(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_emergency(source: Path, destination: Path) -> None:
    os.rename(source, destination)
    _fsync_directory_emergency(destination.parent)


def _copy_database(source: Path, destination: Path) -> None:
    with _connect_read_only(source) as source_connection, sqlite3.connect(destination) as destination_connection:
        source_connection.backup(destination_connection)
    _fsync_file(destination)
    _fsync_directory(destination.parent)


def _copy_database_emergency(source: Path, destination: Path) -> None:
    """Retry snapshot restoration without the normal copy seam after an injected fault."""
    with _connect_read_only(source) as source_connection, sqlite3.connect(destination) as destination_connection:
        source_connection.backup(destination_connection)
    _fsync_file(destination)
    _fsync_directory(destination.parent)


def _copy_read_only_backup(source: Path, directory: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{prefix}-", suffix=".db", dir=directory)
    os.close(descriptor)
    destination = Path(raw_path)
    try:
        _copy_database(source, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if os.path.lexists(candidate):
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise StoreRestoreError("store sidecar rejected")
            candidate.unlink()
    _fsync_directory(path.parent)


def _remove_sidecars_emergency(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if os.path.lexists(candidate):
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise StoreRestoreError("store sidecar rejected")
            candidate.unlink()
    _fsync_directory_emergency(path.parent)
