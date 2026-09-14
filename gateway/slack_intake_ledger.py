"""Small metadata-only durable ledger for Slack intake lifecycle stages.

The ledger stores HMAC digests, fixed labels, and finite timestamps only. Socket
Mode writes run off the event loop and degrade to a bounded in-memory receipt;
the message listener still runs when local diagnostics are unavailable.
"""

from __future__ import annotations

import asyncio
import errno
import hmac
import logging
import math
import os
import re
import secrets
import sqlite3
import stat
import tempfile
import threading
import time
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from weakref import WeakKeyDictionary

from hermes_cli.sqlite_util import open_db, transaction, write_txn
from hermes_constants import get_hermes_home, mkdir_under_hermes_home
from hermes_state_wal import apply_wal_with_fallback
from tools.thread_context import propagate_context_to_thread

_LOGGER = logging.getLogger(__name__)
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_EVENT_TYPES = frozenset({"message", "app_mention"})
_REASONS = frozenset(
    {
        "unauthorized",
        "duplicate_event",
        "duplicate_ts",
        "invalid_event",
        "ignored_channel",
        "bot_message",
        "missing_mention",
        "message_deleted",
        "dm_disabled",
        "listener_cancelled",
        "exception_runtime_error",
        "exception_value_error",
        "exception_type_error",
        "exception_other",
        "not_allowed",
        "ignored",
        "persistence_exhausted",
        "stale_open",
    }
)
_INITIAL_STAGES = {
    "envelope_received": "envelope_duplicate",
    "listener_received": "listener_duplicate",
}
_PROGRESS_STAGES = frozenset(
    {
        "acknowledged",
        "listener_entered",
    }
)
_TERMINAL_STATES = frozenset({"accepted", "dropped", "dead_lettered"})
_ALL_STAGES = frozenset(_INITIAL_STAGES) | frozenset(_INITIAL_STAGES.values()) | _PROGRESS_STAGES | _TERMINAL_STATES

_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 250
_LEDGER_WORK_TIMEOUT = 2.0
_LEDGER_PERMITS = threading.BoundedSemaphore(32)
_LEDGER_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="slack-intake-ledger"
)
_LEDGER_START_LOCK = threading.Lock()
_LEDGER_READY: WeakKeyDictionary[ThreadPoolExecutor, Future] = WeakKeyDictionary()
_DB_LOCK = threading.RLock()
_INIT_LOCK = threading.Lock()
_KEY_LOCK = threading.Lock()
_HEALTH_LOCK = threading.Lock()
_KeyIdentity = tuple[int, int, int, int, int, int]
_KEY_CACHE: dict[str, tuple[_KeyIdentity, bytes]] = {}
_FALLBACK_RECEIPTS: dict[str, list[dict[str, Any]]] = {}
_FALLBACK_EVICTIONS: dict[str, int] = {}
_DEGRADED_HOMES: set[str] = set()
_LOGGED_FAILURES: dict[str, set[str]] = {}
_INITIALIZED_DATABASES: dict[str, tuple[int, int]] = {}

_MAX_IDENTIFIER_LENGTH = 256
_MAX_RECEIPTS = 10_000
_MAX_OPEN_RECEIPTS = 1_000
_MAX_EVENTS_PER_RECEIPT = 64
_MAX_FALLBACK_RECEIPTS = 1_000
_MAX_DATABASE_BYTES = 32 * 1024 * 1024
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_OPEN_AGE_SECONDS = 24 * 60 * 60


class InvalidIntakeTransition(RuntimeError):
    """A receipt was asked to enter a conflicting terminal state."""


class IntakeLedgerCapacityError(RuntimeError):
    """The bounded receipt ledger cannot admit another identity."""


class IntakePersistenceRequired(RuntimeError):
    """HTTP intake cannot acknowledge before its durable receipt exists."""

    def __init__(self, failure_reason: str) -> None:
        self.failure_reason = failure_reason
        super().__init__(
            f"Slack intake receipt is not durable ({failure_reason})"
        )


class _LedgerExecutorUnavailable(RuntimeError):
    """Fixed operational failure at the executor boundary."""


@dataclass(frozen=True)
class ReceiptObservation:
    receipt_id: str
    duplicate: bool
    transport_hash: str
    transport_lookup_hash: str
    message_key_hash: str


@dataclass(frozen=True)
class TerminalObservation:
    state: str
    reason: Optional[str]
    related_receipt_id: Optional[str]


@dataclass(frozen=True)
class StageObservation:
    stage: str
    reason: Optional[str]
    appended: bool


@dataclass(frozen=True)
class PersistenceOutcome:
    persisted: bool
    observation: Optional[Any]
    failure_reason: Optional[str]


def _db_path() -> Path:
    """Resolve the active profile at operation time."""
    return get_hermes_home() / "runtime" / "slack-intake" / "ledger.sqlite3"


def _health_key() -> str:
    return str(_db_path().parent.absolute())


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{name} exceeds the identifier length bound")
    if value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} contains control characters or surrounding whitespace")
    return value


def _validate_label(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_LABEL.fullmatch(value):
        raise ValueError(f"{name} must be a short safe label")
    return value


def _validated_timestamp(value: Any) -> float:
    try:
        timestamp = time.time() if value is None else float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError("intake timestamp must be finite") from None
    if not math.isfinite(timestamp):
        raise ValueError("intake timestamp must be finite")
    return timestamp


def _prepare_private_directory(path: Path) -> None:
    mkdir_under_hermes_home(path)
    try:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
            raise OSError
        if os.name == "posix":
            os.chmod(path, 0o700)
            info = path.stat()
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise OSError
    except OSError:
        raise OSError("Slack intake storage directory is unsafe") from None


def _key_identity(info: os.stat_result) -> _KeyIdentity:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(stat.S_IMODE(info.st_mode)),
        int(info.st_nlink),
    )


def _key_info(path: Path) -> tuple[_KeyIdentity, bytes]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_nlink != 1:
            raise OSError
        if os.name == "posix" and (
            info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise OSError
        key = path.read_bytes()
        if len(key) != 32:
            raise OSError
        after = path.lstat()
        if _key_identity(info) != _key_identity(after):
            raise OSError
    except OSError:
        raise OSError("Slack intake correlation key is unsafe") from None
    return _key_identity(info), key


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("Slack intake storage directory is unsafe")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_key_atomically(path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".correlation-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        key = secrets.token_bytes(32)
        written = os.write(descriptor, key)
        if written != len(key):
            raise OSError("short correlation key write")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if os.name == "posix":
            os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        if os.name == "posix":
            _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _correlation_key() -> bytes:
    path = _db_path().with_name("correlation.key")
    cache_key = str(path.absolute())
    with _KEY_LOCK:
        cached = _KEY_CACHE.get(cache_key)
        if cached is not None:
            try:
                info = path.lstat()
                if cached[0] == _key_identity(info):
                    return cached[1]
            except OSError:
                raise OSError("Slack intake correlation key changed while in use") from None
            raise OSError("Slack intake correlation key changed while in use")
        _prepare_private_directory(path.parent)
        if not path.exists():
            _create_key_atomically(path)
        elif os.name == "posix":
            # A prior creator may have linked the key but failed before its
            # directory entry was durably published. Certify that boundary
            # before allowing the key to identify persisted receipts.
            _fsync_directory(path.parent)
        identity, key = _key_info(path)
        _KEY_CACHE[cache_key] = (identity, key)
        return key


def _digest(namespace: str, value: str, key: Optional[bytes] = None) -> str:
    _validate_label("namespace", namespace)
    return hmac.new(key or _correlation_key(), (namespace + "\0" + value).encode(), "sha256").hexdigest()


def hash_identifier(namespace: str, value: str) -> str:
    _validate_identifier("identifier value", value)
    return _digest(namespace, value)


def _hash_composite(namespace: str, *values: str) -> str:
    for index, value in enumerate(values):
        _validate_identifier(f"identifier {index}", value)
    return _digest(namespace, "\x1f".join(values))


def _receipt_id(workspace_id: str, event_id: str) -> str:
    return _hash_composite("slack-event", workspace_id, event_id)


def _message_key(workspace_id: str, channel_id: str, message_id: str) -> str:
    return _hash_composite("slack-message", workspace_id, channel_id, message_id)


def _transport_key(workspace_id: str, transport_id: str) -> str:
    return _hash_composite("slack-transport", workspace_id, transport_id)


def _cached_digest(namespace: str, value: str) -> Optional[str]:
    path = _db_path().with_name("correlation.key")
    with _KEY_LOCK:
        cached = _KEY_CACHE.get(str(path.absolute()))
    if cached is None:
        return None
    return _digest(namespace, value, cached[1])


def _fallback_digest(namespace: str, value: Any) -> Optional[str]:
    try:
        _validate_identifier("identifier value", value)
        return _cached_digest(namespace, value)
    except (OSError, TypeError, ValueError):
        return None


def _receipts_ddl() -> str:
    return """CREATE TABLE IF NOT EXISTS slack_intake_receipts (
        receipt_id TEXT PRIMARY KEY,
        workspace_hash TEXT NOT NULL,
        event_hash TEXT NOT NULL,
        event_type TEXT NOT NULL,
        channel_hash TEXT NOT NULL,
        thread_hash TEXT,
        message_hash TEXT NOT NULL,
        message_key_hash TEXT NOT NULL,
        first_received_at REAL NOT NULL,
        last_received_at REAL NOT NULL,
        receive_count INTEGER NOT NULL DEFAULT 1,
        terminal_state TEXT,
        terminal_reason TEXT,
        related_receipt_id TEXT,
        decided_at REAL
    )"""


_EXPECTED_SCHEMA = {
    "slack_intake_receipts": (
        ("receipt_id", "TEXT", 0, 1),
        ("workspace_hash", "TEXT", 1, 0),
        ("event_hash", "TEXT", 1, 0),
        ("event_type", "TEXT", 1, 0),
        ("channel_hash", "TEXT", 1, 0),
        ("thread_hash", "TEXT", 0, 0),
        ("message_hash", "TEXT", 1, 0),
        ("message_key_hash", "TEXT", 1, 0),
        ("first_received_at", "REAL", 1, 0),
        ("last_received_at", "REAL", 1, 0),
        ("receive_count", "INTEGER", 1, 0),
        ("terminal_state", "TEXT", 0, 0),
        ("terminal_reason", "TEXT", 0, 0),
        ("related_receipt_id", "TEXT", 0, 0),
        ("decided_at", "REAL", 0, 0),
    ),
    "slack_message_projections": (
        ("message_key_hash", "TEXT", 0, 1),
        ("accepted_receipt_id", "TEXT", 1, 0),
        ("accepted_at", "REAL", 1, 0),
    ),
    "slack_intake_events": (
        ("sequence", "INTEGER", 0, 1),
        ("receipt_id", "TEXT", 1, 0),
        ("stage", "TEXT", 1, 0),
        ("reason", "TEXT", 0, 0),
        ("transport_hash", "TEXT", 0, 0),
        ("related_receipt_id", "TEXT", 0, 0),
        ("observed_at", "REAL", 1, 0),
    ),
}


def _schema_mismatch(detail: str) -> sqlite3.DatabaseError:
    return sqlite3.DatabaseError(f"Slack intake schema mismatch: {detail}")


def _validate_schema(conn: sqlite3.Connection) -> None:
    integrity = conn.execute("PRAGMA quick_check").fetchone()
    if not integrity or str(integrity[0]).lower() != "ok":
        raise _schema_mismatch("integrity check failed")
    for table, expected in _EXPECTED_SCHEMA.items():
        columns = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in conn.execute(f"PRAGMA table_info({table})")
        )
        if columns != expected:
            raise _schema_mismatch(f"unexpected {table} columns")
    foreign_keys = conn.execute("PRAGMA foreign_key_list(slack_intake_events)").fetchall()
    if len(foreign_keys) != 1 or (
        str(foreign_keys[0][2]),
        str(foreign_keys[0][3]),
        str(foreign_keys[0][4]),
        str(foreign_keys[0][6]).upper(),
    ) != ("slack_intake_receipts", "receipt_id", "receipt_id", "CASCADE"):
        raise _schema_mismatch("unexpected intake-event foreign key")
    indexes = {
        str(row[1]): (int(row[2]), int(row[4]))
        for row in conn.execute("PRAGMA index_list(slack_intake_events)")
    }
    index_shape = indexes.get("slack_intake_events_receipt_sequence")
    index_columns = tuple(
        str(row[2])
        for row in conn.execute(
            "PRAGMA index_info(slack_intake_events_receipt_sequence)"
        )
    )
    if index_shape != (0, 0) or index_columns != ("receipt_id", "sequence"):
        raise _schema_mismatch("required intake-event index is absent")


def _initialize_schema(conn: sqlite3.Connection) -> None:
    with write_txn(conn):
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row else 0
        if version > _SCHEMA_VERSION:
            raise sqlite3.DatabaseError("Slack intake schema is newer than supported")
        if version == 0:
            existing = {
                str(item[0])
                for item in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if existing & set(_EXPECTED_SCHEMA):
                raise _schema_mismatch("unversioned intake tables already exist")
            conn.execute(_receipts_ddl())
            conn.execute(
                """CREATE TABLE slack_message_projections (
                    message_key_hash TEXT PRIMARY KEY,
                    accepted_receipt_id TEXT NOT NULL,
                    accepted_at REAL NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE slack_intake_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    reason TEXT,
                    transport_hash TEXT,
                    related_receipt_id TEXT,
                    observed_at REAL NOT NULL,
                    FOREIGN KEY(receipt_id) REFERENCES slack_intake_receipts(receipt_id) ON DELETE CASCADE
                )"""
            )
            conn.execute(
                "CREATE INDEX slack_intake_events_receipt_sequence "
                "ON slack_intake_events(receipt_id, sequence)"
            )
            conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        _validate_schema(conn)


def _open_connection() -> sqlite3.Connection:
    path = _db_path()
    _prepare_private_directory(path.parent)
    try:
        info = path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_nlink != 1:
            raise OSError("Slack intake database path is unsafe")
    conn = open_db(
        path,
        db_label="Slack intake ledger",
        busy_timeout_ms=_BUSY_TIMEOUT_MS,
        wal=False,
        foreign_keys=True,
        synchronous_full=True,
    )
    try:
        info = path.lstat()
        identity = (int(info.st_dev), int(info.st_ino))
        cache_key = str(path.absolute())
        with _INIT_LOCK:
            if _INITIALIZED_DATABASES.get(cache_key) != identity:
                journal_mode = apply_wal_with_fallback(
                    conn, db_label="Slack intake ledger"
                )
                if not isinstance(journal_mode, str):
                    raise sqlite3.OperationalError(
                        "Slack intake journal mode was not accepted"
                    )
                journal_mode = journal_mode.strip().lower()
                if journal_mode not in {"wal", "delete"}:
                    raise sqlite3.OperationalError(
                        "Slack intake journal mode was not accepted"
                    )
                actual_mode = conn.execute("PRAGMA journal_mode").fetchone()
                actual_mode = (
                    str(actual_mode[0]).strip().lower() if actual_mode else ""
                )
                if actual_mode not in {"wal", "delete"} or actual_mode != journal_mode:
                    raise sqlite3.OperationalError(
                        "Slack intake journal mode could not be verified"
                    )
                _initialize_schema(conn)
                _INITIALIZED_DATABASES[cache_key] = identity
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        max_pages = max(1, _MAX_DATABASE_BYTES // page_size)
        effective = int(
            conn.execute(f"PRAGMA max_page_count={max_pages}").fetchone()[0]
        )
        if effective < page_count:
            raise sqlite3.DatabaseError("Slack intake database size limit is invalid")
    except BaseException:
        conn.close()
        raise
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            conn.close()
            raise OSError("Slack intake ledger file is unsafe") from None
    return conn


def _reset_initialization_for_tests() -> None:
    with _INIT_LOCK:
        _INITIALIZED_DATABASES.clear()
    with _KEY_LOCK:
        _KEY_CACHE.clear()


def _trim_events(conn: sqlite3.Connection, receipt_id: str) -> None:
    count = int(
        conn.execute(
            "SELECT COUNT(*) FROM slack_intake_events WHERE receipt_id=?", (receipt_id,)
        ).fetchone()[0]
    )
    excess = count - _MAX_EVENTS_PER_RECEIPT
    if excess <= 0:
        return
    conn.execute(
        """DELETE FROM slack_intake_events
           WHERE sequence IN (
               SELECT sequence FROM slack_intake_events
               WHERE receipt_id=?
                 AND sequence != (
                     SELECT MIN(sequence) FROM slack_intake_events WHERE receipt_id=?
                 )
                 AND stage NOT IN ('accepted', 'dropped', 'dead_lettered')
               ORDER BY sequence LIMIT ?
           )""",
        (receipt_id, receipt_id, excess),
    )


def _prune(conn: sqlite3.Connection, now: float) -> None:
    stale = now - _MAX_OPEN_AGE_SECONDS
    stale_ids = [
        row[0]
        for row in conn.execute(
            "SELECT receipt_id FROM slack_intake_receipts "
            "WHERE terminal_state IS NULL AND last_received_at < ?",
            (stale,),
        )
    ]
    for receipt_id in stale_ids:
        conn.execute(
            "UPDATE slack_intake_receipts SET terminal_state='dead_lettered', "
            "terminal_reason='stale_open', decided_at=? WHERE receipt_id=?",
            (now, receipt_id),
        )
        conn.execute(
            "INSERT INTO slack_intake_events(receipt_id, stage, reason, observed_at) "
            "VALUES (?, 'dead_lettered', 'stale_open', ?)",
            (receipt_id, now),
        )
        _trim_events(conn, receipt_id)
    cutoff = now - _RETENTION_SECONDS
    conn.execute(
        "DELETE FROM slack_message_projections WHERE accepted_receipt_id IN ("
        "SELECT receipt_id FROM slack_intake_receipts WHERE decided_at < ?)",
        (cutoff,),
    )
    conn.execute("DELETE FROM slack_intake_receipts WHERE decided_at < ?", (cutoff,))


def _occupied_database_bytes(conn: sqlite3.Connection) -> int:
    page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
    freelist_count = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    if page_count < 0 or freelist_count < 0 or page_size <= 0:
        raise sqlite3.DatabaseError("Slack intake database size is invalid")
    return max(0, page_count - freelist_count) * page_size


def record_listener_received(
    *,
    workspace_id: str,
    event_id: str,
    transport_id: str,
    event_type: str,
    channel_id: str,
    thread_id: Optional[str],
    message_id: str,
    received_at: Optional[float] = None,
    stage: str = "listener_received",
) -> ReceiptObservation:
    for name, value in (
        ("workspace_id", workspace_id),
        ("event_id", event_id),
        ("transport_id", transport_id),
        ("channel_id", channel_id),
        ("message_id", message_id),
    ):
        _validate_identifier(name, value)
    if thread_id is not None:
        _validate_identifier("thread_id", thread_id)
    if not isinstance(event_type, str) or event_type not in _EVENT_TYPES:
        raise ValueError("event_type must be a fixed intake event type")
    if not isinstance(stage, str) or stage not in _INITIAL_STAGES:
        raise ValueError("stage must be an initial intake stage")
    observed_at = _validated_timestamp(received_at)
    receipt_id = _receipt_id(workspace_id, event_id)
    message_key_hash = _message_key(workspace_id, channel_id, message_id)
    transport_hash = _transport_key(workspace_id, transport_id)
    transport_lookup_hash = hash_identifier("transport", transport_id)
    expected = (
        hash_identifier("workspace", workspace_id),
        hash_identifier("event", event_id),
        event_type,
        hash_identifier("channel", channel_id),
        hash_identifier("thread", thread_id) if thread_id else None,
        hash_identifier("message", message_id),
        message_key_hash,
    )

    with _DB_LOCK, transaction(_open_connection(), immediate=True) as conn:
        existing = conn.execute(
            "SELECT workspace_hash, event_hash, event_type, channel_hash, thread_hash, "
            "message_hash, message_key_hash FROM slack_intake_receipts WHERE receipt_id=?",
            (receipt_id,),
        ).fetchone()
        duplicate = existing is not None
        event_stage = stage
        if existing is None:
            _prune(conn, observed_at)
            if _occupied_database_bytes(conn) >= _MAX_DATABASE_BYTES:
                raise IntakeLedgerCapacityError("Slack intake ledger capacity reached")
            receipt_count = int(conn.execute("SELECT COUNT(*) FROM slack_intake_receipts").fetchone()[0])
            open_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM slack_intake_receipts WHERE terminal_state IS NULL"
                ).fetchone()[0]
            )
            if receipt_count >= _MAX_RECEIPTS or open_count >= _MAX_OPEN_RECEIPTS:
                raise IntakeLedgerCapacityError("Slack intake ledger capacity reached")
            conn.execute(
                """INSERT INTO slack_intake_receipts (
                    receipt_id, workspace_hash, event_hash, event_type, channel_hash,
                    thread_hash, message_hash, message_key_hash, first_received_at,
                    last_received_at, receive_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (receipt_id, *expected, observed_at, observed_at),
            )
        elif tuple(existing) != expected:
            raise InvalidIntakeTransition("receipt identity metadata changed")
        else:
            first = conn.execute(
                "SELECT stage FROM slack_intake_events WHERE receipt_id=? ORDER BY sequence LIMIT 1",
                (receipt_id,),
            ).fetchone()
            if first is None or first[0] not in _INITIAL_STAGES:
                raise InvalidIntakeTransition("receipt initial stage is invalid")
            event_stage = _INITIAL_STAGES[first[0]]
            conn.execute(
                "UPDATE slack_intake_receipts SET last_received_at=MAX(last_received_at, ?), "
                "receive_count=receive_count+1 WHERE receipt_id=?",
                (observed_at, receipt_id),
            )
        conn.execute(
            "INSERT INTO slack_intake_events(receipt_id, stage, transport_hash, observed_at) "
            "VALUES (?, ?, ?, ?)",
            (receipt_id, event_stage, transport_hash, observed_at),
        )
        _trim_events(conn, receipt_id)
    return ReceiptObservation(
        receipt_id=receipt_id,
        duplicate=duplicate,
        transport_hash=transport_hash,
        transport_lookup_hash=transport_lookup_hash,
        message_key_hash=message_key_hash,
    )


def append_stage(
    receipt_id: str,
    *,
    stage: str,
    observed_at: Optional[float] = None,
    reason: Optional[str] = None,
) -> StageObservation:
    if not isinstance(receipt_id, str) or not _DIGEST.fullmatch(receipt_id):
        raise ValueError("receipt_id must be a 64-character digest")
    if not isinstance(stage, str) or stage not in _PROGRESS_STAGES:
        raise ValueError("stage must be a fixed progress stage")
    if reason is not None and (
        not isinstance(reason, str) or reason not in _REASONS
    ):
        raise ValueError("reason must be a fixed intake reason")
    timestamp = _validated_timestamp(observed_at)
    with _DB_LOCK, transaction(_open_connection(), immediate=True) as conn:
        if conn.execute(
            "SELECT 1 FROM slack_intake_receipts WHERE receipt_id=?", (receipt_id,)
        ).fetchone() is None:
            raise KeyError(receipt_id)
        latest = conn.execute(
            "SELECT stage, reason FROM slack_intake_events "
            "WHERE receipt_id=? ORDER BY sequence DESC LIMIT 1",
            (receipt_id,),
        ).fetchone()
        if latest is not None and tuple(latest) == (stage, reason):
            return StageObservation(stage, reason, False)
        conn.execute(
            "INSERT INTO slack_intake_events(receipt_id, stage, reason, observed_at) "
            "VALUES (?, ?, ?, ?)",
            (receipt_id, stage, reason, timestamp),
        )
        _trim_events(conn, receipt_id)
    return StageObservation(stage, reason, True)


def _mark_terminal(
    receipt_id: str,
    *,
    state: str,
    reason: Optional[str],
    decided_at: Optional[float],
) -> TerminalObservation:
    if not isinstance(receipt_id, str) or not _DIGEST.fullmatch(receipt_id):
        raise ValueError("receipt_id must be a 64-character digest")
    if not isinstance(state, str) or state not in _TERMINAL_STATES:
        raise ValueError("invalid terminal state")
    if state == "accepted" and reason is not None:
        raise ValueError("accepted receipts cannot carry a reason")
    if state != "accepted" and (
        not isinstance(reason, str) or reason not in _REASONS
    ):
        raise ValueError("drop reason must be a fixed intake reason")
    timestamp = _validated_timestamp(decided_at)

    with _DB_LOCK, transaction(_open_connection(), immediate=True) as conn:
        row = conn.execute(
            "SELECT terminal_state, terminal_reason, related_receipt_id, message_key_hash "
            "FROM slack_intake_receipts WHERE receipt_id=?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(receipt_id)
        current_state, current_reason, current_related, message_key_hash = row
        if current_state is not None:
            if current_state == state and current_reason == reason:
                return TerminalObservation(current_state, current_reason, current_related)
            if current_state == "accepted" and state == "dropped" and reason == "duplicate_event":
                return TerminalObservation(current_state, current_reason, current_related)
            if current_state == "dropped" and current_reason == "duplicate_ts" and state == "accepted":
                return TerminalObservation(current_state, current_reason, current_related)
            raise InvalidIntakeTransition(f"receipt already terminal as {current_state}")

        related = None
        if state == "accepted":
            sibling = conn.execute(
                "SELECT accepted_receipt_id FROM slack_message_projections WHERE message_key_hash=?",
                (message_key_hash,),
            ).fetchone()
            if sibling is None:
                conn.execute(
                    "INSERT INTO slack_message_projections(message_key_hash, accepted_receipt_id, accepted_at) "
                    "VALUES (?, ?, ?)",
                    (message_key_hash, receipt_id, timestamp),
                )
            elif sibling[0] != receipt_id:
                state, reason, related = "dropped", "duplicate_ts", sibling[0]
        elif reason == "duplicate_event":
            sibling = conn.execute(
                "SELECT accepted_receipt_id FROM slack_message_projections WHERE message_key_hash=?",
                (message_key_hash,),
            ).fetchone()
            if sibling is not None and sibling[0] != receipt_id:
                reason, related = "duplicate_ts", sibling[0]

        conn.execute(
            "UPDATE slack_intake_receipts SET terminal_state=?, terminal_reason=?, "
            "related_receipt_id=?, decided_at=? WHERE receipt_id=?",
            (state, reason, related, timestamp, receipt_id),
        )
        conn.execute(
            "INSERT INTO slack_intake_events(receipt_id, stage, reason, related_receipt_id, observed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (receipt_id, state, reason, related, timestamp),
        )
        _trim_events(conn, receipt_id)
    return TerminalObservation(state, reason, related)


def mark_accepted(receipt_id: str, *, decided_at: Optional[float] = None) -> TerminalObservation:
    return _mark_terminal(receipt_id, state="accepted", reason=None, decided_at=decided_at)


def mark_dropped(
    receipt_id: str, *, reason: str, decided_at: Optional[float] = None
) -> TerminalObservation:
    return _mark_terminal(receipt_id, state="dropped", reason=reason, decided_at=decided_at)


def mark_dead_lettered(
    receipt_id: str, *, reason: str, decided_at: Optional[float] = None
) -> TerminalObservation:
    return _mark_terminal(receipt_id, state="dead_lettered", reason=reason, decided_at=decided_at)


def _ledger_worker_ready(executor: ThreadPoolExecutor) -> Future:
    """Certify the single worker before submitting side-effecting work."""
    global _LEDGER_EXECUTOR
    if not _LEDGER_START_LOCK.acquire(blocking=False):
        raise TimeoutError("intake worker startup busy")
    try:
        if executor._max_workers != 1:
            raise _LedgerExecutorUnavailable("intake executor unavailable")
        if executor in _LEDGER_READY:
            return _LEDGER_READY[executor]
        try:
            ready = executor.submit(lambda: None)
        except RuntimeError:
            if not executor._shutdown:
                executor.shutdown(wait=False, cancel_futures=True)
                if _LEDGER_EXECUTOR is executor:
                    _LEDGER_EXECUTOR = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=(
                            executor._thread_name_prefix or "slack-intake-ledger"
                        ),
                    )
            raise _LedgerExecutorUnavailable("intake executor unavailable") from None
        _LEDGER_READY[executor] = ready
        return ready
    finally:
        _LEDGER_START_LOCK.release()


async def _await_ledger_future(future: Future, deadline: float) -> Any:
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(
        lambda done: None if done.cancelled() else done.exception()
    )
    try:
        return await asyncio.wait_for(
            asyncio.shield(wrapped), max(0.0, deadline - time.monotonic())
        )
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if future.cancelled() and task is not None and not task.cancelling():
            raise _LedgerExecutorUnavailable("intake executor unavailable") from None
        raise


async def _run_ledger_work(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one bounded SQLite call on a certified dedicated worker."""
    deadline = time.monotonic() + _LEDGER_WORK_TIMEOUT
    permits = _LEDGER_PERMITS
    if not permits.acquire(blocking=False):
        raise TimeoutError("intake work admission deadline exceeded")
    executor = _LEDGER_EXECUTOR
    try:
        ready = _ledger_worker_ready(executor)
    except BaseException:
        permits.release()
        raise
    try:
        if not ready.done():
            await _await_ledger_future(ready, deadline)
        ready.result()
    except (RuntimeError, FutureCancelledError):
        ready.add_done_callback(lambda _done: permits.release())
        raise _LedgerExecutorUnavailable("intake executor unavailable") from None
    except BaseException:
        ready.add_done_callback(lambda _done: permits.release())
        raise

    def work() -> Any:
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("intake queued work deadline exceeded")
            return function(*args, **kwargs)
        finally:
            permits.release()

    try:
        future = executor.submit(propagate_context_to_thread(work))
    except RuntimeError:
        permits.release()
        raise _LedgerExecutorUnavailable("intake executor unavailable") from None
    except BaseException:
        permits.release()
        raise
    future.add_done_callback(
        lambda done: permits.release() if done.cancelled() else None
    )
    return await _await_ledger_future(future, deadline)


def _classify_persistence_failure(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "work_deadline_exceeded"
    if isinstance(exc, IntakeLedgerCapacityError):
        return "capacity_exceeded"
    if isinstance(exc, InvalidIntakeTransition):
        return "identity_conflict"
    if isinstance(exc, (TypeError, ValueError)):
        return "invalid_metadata"
    if isinstance(exc, OSError):
        if exc.errno == errno.EDQUOT:
            return "quota_exceeded"
        if exc.errno == errno.ENOSPC:
            return "disk_full"
    code = getattr(exc, "sqlite_errorcode", None)
    base_code = code & 0xFF if isinstance(code, int) else None
    message = str(exc).lower()
    if base_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or "database is locked" in message:
        return "busy_deadline_exceeded"
    if base_code == sqlite3.SQLITE_FULL or "disk is full" in message:
        return "disk_full"
    if base_code == sqlite3.SQLITE_CORRUPT or any(
        marker in message for marker in ("malformed", "not a database", "corrupt")
    ):
        return "database_corrupt"
    if "newer than supported" in message or "slack intake schema mismatch" in message:
        return "schema_mismatch"
    return "persistence_failed"


def _fallback_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return _validated_timestamp(value)
        except ValueError:
            pass
    return time.time()


def _fallback_receipt_id(workspace_id: Any, event_id: Any) -> Optional[str]:
    try:
        _validate_identifier("workspace_id", workspace_id)
        _validate_identifier("event_id", event_id)
        return _cached_digest("slack-event", workspace_id + "\x1f" + event_id)
    except (OSError, TypeError, ValueError):
        return None


def _buffer_fallback_entry(entry: dict[str, Any]) -> None:
    key = _health_key()
    with _HEALTH_LOCK:
        receipts = _FALLBACK_RECEIPTS.setdefault(key, [])
        if len(receipts) >= _MAX_FALLBACK_RECEIPTS:
            del receipts[0]
            _FALLBACK_EVICTIONS[key] = _FALLBACK_EVICTIONS.get(key, 0) + 1
        receipts.append(entry)


def _buffer_fallback_receipt(kwargs: dict[str, Any], failure_reason: str) -> None:
    event_type = kwargs.get("event_type")
    if not isinstance(event_type, str) or event_type not in _EVENT_TYPES:
        event_type = "unknown"
    stage = kwargs.get("stage", "listener_received")
    _buffer_fallback_entry(
        {
            "receipt_id": _fallback_receipt_id(kwargs.get("workspace_id"), kwargs.get("event_id")),
            "workspace_hash": _fallback_digest("workspace", kwargs.get("workspace_id")),
            "event_hash": _fallback_digest("event", kwargs.get("event_id")),
            "transport_hash": _fallback_digest("transport", kwargs.get("transport_id")),
            "event_type": event_type,
            "stage": (
                stage
                if isinstance(stage, str) and stage in _INITIAL_STAGES
                else "unknown"
            ),
            "failure_reason": failure_reason,
            "observed_at": _fallback_timestamp(kwargs.get("received_at")),
        }
    )


def _note_persistence_failed(failure_reason: str, exc: BaseException) -> None:
    key = _health_key()
    with _HEALTH_LOCK:
        _DEGRADED_HOMES.add(key)
        logged = _LOGGED_FAILURES.setdefault(key, set())
        if failure_reason in logged:
            return
        logged.add(failure_reason)
    _LOGGER.critical(
        "Slack intake persistence failed (reason=%s, error_type=%s); bounded fallback active",
        failure_reason,
        type(exc).__name__,
    )


def _note_persistence_recovered() -> None:
    key = _health_key()
    with _HEALTH_LOCK:
        if key not in _DEGRADED_HOMES:
            return
        _DEGRADED_HOMES.discard(key)
        _LOGGED_FAILURES.pop(key, None)
    _LOGGER.warning("Slack intake ledger persistence recovered")


def record_unavailable_envelope(**kwargs: Any) -> None:
    """Retain bounded metadata when local admission is already occupied."""
    metadata = dict(kwargs)
    metadata["stage"] = "envelope_received"
    _buffer_fallback_receipt(metadata, "admission_busy")
    _note_persistence_failed("admission_busy", RuntimeError())


def record_unavailable_stage(
    receipt_id: Optional[str],
    *,
    stage: str,
    observed_at: Optional[float],
    reason: Optional[str] = None,
) -> None:
    """Retain a bounded stage marker when local admission is occupied."""
    _buffer_fallback_entry(
        {
            "receipt_id": (
                receipt_id
                if isinstance(receipt_id, str) and _DIGEST.fullmatch(receipt_id)
                else None
            ),
            "stage": (
                stage
                if isinstance(stage, str)
                and stage in _PROGRESS_STAGES | _TERMINAL_STATES
                else "unknown"
            ),
            "reason": reason if isinstance(reason, str) and reason in _REASONS else None,
            "failure_reason": "admission_busy",
            "observed_at": _fallback_timestamp(observed_at),
        }
    )
    _note_persistence_failed("admission_busy", RuntimeError())


def persistence_degraded() -> bool:
    with _HEALTH_LOCK:
        return _health_key() in _DEGRADED_HOMES


def read_fallback_receipts() -> list[dict[str, Any]]:
    with _HEALTH_LOCK:
        return [dict(item) for item in _FALLBACK_RECEIPTS.get(_health_key(), [])]


def fallback_eviction_count() -> int:
    with _HEALTH_LOCK:
        return _FALLBACK_EVICTIONS.get(_health_key(), 0)


def _reset_persistence_health_for_tests() -> None:
    with _HEALTH_LOCK:
        _FALLBACK_RECEIPTS.clear()
        _FALLBACK_EVICTIONS.clear()
        _DEGRADED_HOMES.clear()
        _LOGGED_FAILURES.clear()


_SAFE_FAILURE_TYPES = (
    sqlite3.Error,
    OSError,
    RuntimeError,
    IntakeLedgerCapacityError,
    InvalidIntakeTransition,
    TypeError,
    ValueError,
)


async def record_listener_received_safely(**kwargs: Any) -> PersistenceOutcome:
    try:
        observation = await _run_ledger_work(record_listener_received, **kwargs)
    except asyncio.CancelledError:
        raise
    except _SAFE_FAILURE_TYPES as exc:
        failure_reason = _classify_persistence_failure(exc)
        _buffer_fallback_receipt(kwargs, failure_reason)
        _note_persistence_failed(failure_reason, exc)
        return PersistenceOutcome(False, None, failure_reason)
    _note_persistence_recovered()
    return PersistenceOutcome(True, observation, None)


async def record_http_envelope(**kwargs: Any) -> ReceiptObservation:
    """Persist HTTP ingress before its caller is allowed to return success."""
    metadata = dict(kwargs)
    metadata["stage"] = "envelope_received"
    try:
        observation = await _run_ledger_work(record_listener_received, **metadata)
    except asyncio.CancelledError:
        raise
    except _SAFE_FAILURE_TYPES as exc:
        failure_reason = _classify_persistence_failure(exc)
        _note_persistence_failed(failure_reason, exc)
        raise IntakePersistenceRequired(failure_reason) from None
    _note_persistence_recovered()
    return observation


async def append_stage_safely(
    receipt_id: str,
    *,
    stage: str,
    observed_at: Optional[float] = None,
    reason: Optional[str] = None,
) -> PersistenceOutcome:
    try:
        observation = await _run_ledger_work(
            append_stage,
            receipt_id,
            stage=stage,
            observed_at=observed_at,
            reason=reason,
        )
    except asyncio.CancelledError:
        raise
    except _SAFE_FAILURE_TYPES + (KeyError,) as exc:
        failure_reason = "unknown_receipt" if isinstance(exc, KeyError) else _classify_persistence_failure(exc)
        _buffer_fallback_entry(
            {
                "receipt_id": receipt_id if isinstance(receipt_id, str) and _DIGEST.fullmatch(receipt_id) else None,
                "stage": (
                    stage
                    if isinstance(stage, str) and stage in _PROGRESS_STAGES
                    else "unknown"
                ),
                "reason": (
                    reason
                    if isinstance(reason, str) and reason in _REASONS
                    else None
                ),
                "failure_reason": failure_reason,
                "observed_at": _fallback_timestamp(observed_at),
            }
        )
        _note_persistence_failed(failure_reason, exc)
        return PersistenceOutcome(False, None, failure_reason)
    _note_persistence_recovered()
    return PersistenceOutcome(True, observation, None)


async def _mark_terminal_safely(
    receipt_id: str, *, state: str, reason: Optional[str], decided_at: float
) -> PersistenceOutcome:
    function = mark_accepted if state == "accepted" else mark_dropped
    call = {"decided_at": decided_at}
    if reason is not None:
        call["reason"] = reason
    try:
        observation = await _run_ledger_work(function, receipt_id, **call)
    except asyncio.CancelledError:
        raise
    except _SAFE_FAILURE_TYPES + (KeyError,) as exc:
        failure_reason = "unknown_receipt" if isinstance(exc, KeyError) else _classify_persistence_failure(exc)
        _buffer_fallback_entry(
            {
                "receipt_id": receipt_id if isinstance(receipt_id, str) and _DIGEST.fullmatch(receipt_id) else None,
                "stage": state,
                "reason": (
                    reason
                    if isinstance(reason, str) and reason in _REASONS
                    else None
                ),
                "failure_reason": failure_reason,
                "observed_at": _fallback_timestamp(decided_at),
            }
        )
        _note_persistence_failed(failure_reason, exc)
        return PersistenceOutcome(False, None, failure_reason)
    _note_persistence_recovered()
    return PersistenceOutcome(True, observation, None)


async def mark_accepted_safely(receipt_id: str, *, decided_at: float) -> PersistenceOutcome:
    return await _mark_terminal_safely(
        receipt_id, state="accepted", reason=None, decided_at=decided_at
    )


async def mark_dropped_safely(
    receipt_id: str, *, reason: str, decided_at: float
) -> PersistenceOutcome:
    return await _mark_terminal_safely(
        receipt_id, state="dropped", reason=reason, decided_at=decided_at
    )


def read_receipts(limit: int = 100) -> list[dict[str, Any]]:
    """Return a bounded metadata projection for local diagnostics and tests."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    path = _db_path()
    if not path.is_file() or path.is_symlink():
        return []
    try:
        conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True, timeout=0.25)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM slack_intake_receipts ORDER BY first_received_at, receipt_id LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["events"] = [
                dict(event)
                for event in conn.execute(
                    "SELECT stage, reason, transport_hash, related_receipt_id, observed_at "
                    "FROM slack_intake_events WHERE receipt_id=? ORDER BY sequence",
                    (item["receipt_id"],),
                ).fetchall()
            ]
            result.append(item)
        return result
    except (OSError, sqlite3.Error):
        return []
    finally:
        if "conn" in locals():
            conn.close()
