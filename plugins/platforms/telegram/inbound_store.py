"""Durable Telegram ingress and dispatch primitives.

The Telegram updater advances its polling offset immediately after
``Application.update_queue.put`` returns.  This module supplies the queue used
by the adapter so actionable updates are persisted before that boundary.  The
SQLite schema deliberately stores both bot and update identifiers as TEXT:
Telegram identity is not allowed to narrow through SQLite's signed INTEGER
range.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from hermes_cli.config import get_hermes_home


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_ATTEMPTS = 3
_INGRESS_RETRY_INITIAL_SECONDS = 0.05
_INGRESS_RETRY_MAX_SECONDS = 2.0
_INGRESS_STALL_LOG_SECONDS = 30.0
_DECIMAL_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_TERMINAL_STATES = frozenset({"consumed", "rejected", "dead_letter"})
_STORE_EXECUTOR_LOCK = threading.Lock()
_STORE_EXECUTORS: dict[str, tuple[ThreadPoolExecutor, int]] = {}


def _canonical_store_path(path: os.PathLike[str] | str) -> str:
    """Resolve aliases to one stable process-local SQLite identity."""
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _acquire_store_executor(key: str, bot_account_id: int) -> ThreadPoolExecutor:
    """Share one SQLite worker across queue epochs for one durable inbox."""
    with _STORE_EXECUTOR_LOCK:
        current = _STORE_EXECUTORS.get(key)
        if current is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"telegram-inbound-{bot_account_id}",
            )
            _STORE_EXECUTORS[key] = (executor, 1)
            return executor
        executor, references = current
        _STORE_EXECUTORS[key] = (executor, references + 1)
        return executor


def _release_store_executor(key: str, executor: ThreadPoolExecutor) -> None:
    """Release one queue epoch without interrupting a replacement's writes."""
    should_shutdown = False
    with _STORE_EXECUTOR_LOCK:
        current = _STORE_EXECUTORS.get(key)
        if current is None or current[0] is not executor:
            return
        references = current[1] - 1
        if references > 0:
            _STORE_EXECUTORS[key] = (executor, references)
        else:
            _STORE_EXECUTORS.pop(key, None)
            should_shutdown = True
    if should_shutdown:
        executor.shutdown(wait=False, cancel_futures=False)


def _canonical_decimal(value: Any, name: str) -> str:
    """Return a decimal integer in canonical text form."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and _DECIMAL_INTEGER.fullmatch(value.strip()):
        return str(int(value.strip(), 10))
    raise TypeError(f"{name} must be a decimal integer")


def canonical_bot_account_id(value: Any) -> str:
    """Canonicalize a Telegram bot account id without SQLite narrowing."""
    return _canonical_decimal(value, "bot_account_id")


def bot_account_key(value: Any) -> str:
    """Return the stable key used by account-scoped Telegram state."""
    return f"telegram-account:{canonical_bot_account_id(value)}"


def bot_account_id_from_token(token: Any) -> str:
    """Extract the numeric Telegram bot id from a Bot API token.

    Only the token prefix is used; the secret portion is never persisted or
    included in an exception raised by this function.
    """
    raw = str(token or "")
    prefix = raw.split(":", 1)[0]
    return canonical_bot_account_id(prefix)


@dataclass
class CaptureDecision:
    """Side-effect-free classification of one incoming Telegram update."""

    actionable: bool
    profile: str = "default"
    account_id: str = "telegram"
    update_kind: str = "message"
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    callback_query_id: Optional[str] = None
    session_key: str = ""
    priority: int = 100
    payload: Optional[dict[str, Any]] = None
    receipt_required: bool = False


@dataclass
class InboundEvent:
    """Durable representation of one accepted Telegram update."""

    event_id: str
    bot_account_id: int
    update_id: int
    profile: str
    account_id: str
    update_kind: str
    chat_id: Optional[str]
    message_id: Optional[str]
    callback_query_id: Optional[str]
    session_key: str
    priority: int
    payload: Optional[dict[str, Any]]
    work_state: str
    dispatch_state: str
    received_at: float
    persisted_at: float
    queued_at: Optional[float]
    lease_owner: Optional[str] = None
    lease_epoch: int = 0
    lease_expires_at: Optional[float] = None
    leased_at: Optional[float] = None
    attempt_count: int = 0
    replay_count: int = 0
    last_replay_at: Optional[float] = None
    last_error_class: Optional[str] = None
    duplicate_count: int = 0
    identity_conflict_count: int = 0
    last_duplicate_at: Optional[float] = None
    consumed_at: Optional[float] = None
    terminal_at: Optional[float] = None
    terminal_reason: Optional[str] = None
    payload_sha256: Optional[str] = None
    replayed: bool = False


@dataclass
class PersistResult:
    event_id: str
    duplicate: bool
    identity_conflict: bool
    row: InboundEvent


class TelegramInboundStore:
    """SQLite inbox with transactional, account-scoped ownership changes."""

    def __init__(self, path: Optional[Path] = None, *, busy_timeout_ms: int = 250):
        self.path = Path(path or (get_hermes_home() / "telegram_inbound.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        self._secure_store_files()
        self.lifecycle_path = _canonical_store_path(self.path)
        self._ensure_schema()

    def _secure_store_files(self) -> None:
        """Keep the SQLite database and journaling sidecars owner-readable."""
        for candidate in (
            self.path,
            self.path.with_name(self.path.name + "-wal"),
            self.path.with_name(self.path.name + "-shm"),
        ):
            try:
                file_stat = os.lstat(candidate)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(file_stat.st_mode):
                raise RuntimeError(
                    f"Telegram inbound SQLite path must not be a symlink: {candidate}"
                )
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(
                    f"Telegram inbound SQLite path must be a regular file: {candidate}"
                )
            try:
                os.chmod(candidate, 0o600)
            except FileNotFoundError:
                continue

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )
        self._secure_store_files()
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                existing_tables = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name IN ('telegram_inbound_event', 'telegram_inbound_alias')"
                    ).fetchall()
                }
                if existing_tables == {
                    "telegram_inbound_event",
                    "telegram_inbound_alias",
                }:
                    existing_event_types = {
                        str(row[1]): str(row[2]).upper()
                        for row in conn.execute(
                            "PRAGMA table_info(telegram_inbound_event)"
                        ).fetchall()
                    }
                    existing_alias_types = {
                        str(row[1]): str(row[2]).upper()
                        for row in conn.execute(
                            "PRAGMA table_info(telegram_inbound_alias)"
                        ).fetchall()
                    }
                    if (
                        existing_event_types.get("bot_account_id") != "TEXT"
                        or existing_event_types.get("update_id") != "TEXT"
                        or existing_alias_types.get("bot_account_id") != "TEXT"
                    ):
                        self._validate_identifier_migration_preflight(conn)
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_inbound_event (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL UNIQUE,
                        bot_account_id TEXT NOT NULL,
                        update_id TEXT NOT NULL,
                        profile TEXT NOT NULL,
                        account_id TEXT NOT NULL,
                        update_kind TEXT NOT NULL,
                        chat_id TEXT,
                        message_id TEXT,
                        callback_query_id TEXT,
                        session_key TEXT NOT NULL,
                        priority INTEGER NOT NULL DEFAULT 100,
                        payload_json TEXT,
                        payload_sha256 TEXT,
                        work_state TEXT NOT NULL DEFAULT 'queued',
                        receipt_state TEXT NOT NULL DEFAULT 'not_required',
                        dispatch_state TEXT NOT NULL DEFAULT 'pending',
                        next_attempt_at REAL,
                        received_at REAL NOT NULL,
                        persisted_at REAL NOT NULL,
                        queued_at REAL,
                        lease_owner TEXT,
                        lease_epoch INTEGER NOT NULL DEFAULT 0,
                        lease_expires_at REAL,
                        leased_at REAL,
                        context_committed_at REAL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        replay_count INTEGER NOT NULL DEFAULT 0,
                        last_replay_at REAL,
                        last_error_class TEXT,
                        duplicate_count INTEGER NOT NULL DEFAULT 0,
                        identity_conflict_count INTEGER NOT NULL DEFAULT 0,
                        last_duplicate_at REAL,
                        consumed_at REAL,
                        terminal_at REAL,
                        receipt_attempted_at REAL,
                        receipt_confirmed_at REAL,
                        terminal_reason TEXT
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        telegram_inbound_event_account_update
                        ON telegram_inbound_event(bot_account_id, update_id);
                    CREATE INDEX IF NOT EXISTS telegram_inbound_event_ready
                        ON telegram_inbound_event(bot_account_id, work_state,
                                                  dispatch_state, priority, seq);
                    CREATE TABLE IF NOT EXISTS telegram_inbound_alias (
                        bot_account_id TEXT NOT NULL,
                        alias_kind TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        alias_value TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        PRIMARY KEY (bot_account_id, alias_kind, scope, alias_value),
                        FOREIGN KEY (event_id) REFERENCES telegram_inbound_event(event_id)
                    );
                    CREATE INDEX IF NOT EXISTS telegram_inbound_alias_event
                        ON telegram_inbound_alias(event_id);
                    """
                )
                columns = {
                    str(row[1])
                    for row in conn.execute(
                        "PRAGMA table_info(telegram_inbound_event)"
                    ).fetchall()
                }
                if "next_attempt_at" not in columns:
                    conn.execute(
                        "ALTER TABLE telegram_inbound_event ADD COLUMN next_attempt_at REAL"
                    )
                if "receipt_state" not in columns:
                    conn.execute(
                        "ALTER TABLE telegram_inbound_event "
                        "ADD COLUMN receipt_state TEXT NOT NULL DEFAULT 'not_required'"
                    )
                conn.commit()
                event_types = {
                    str(row[1]): str(row[2]).upper()
                    for row in conn.execute(
                        "PRAGMA table_info(telegram_inbound_event)"
                    ).fetchall()
                }
                alias_types = {
                    str(row[1]): str(row[2]).upper()
                    for row in conn.execute(
                        "PRAGMA table_info(telegram_inbound_alias)"
                    ).fetchall()
                }
                if (
                    event_types.get("bot_account_id") != "TEXT"
                    or event_types.get("update_id") != "TEXT"
                    or alias_types.get("bot_account_id") != "TEXT"
                ):
                    self._migrate_identifier_affinity(conn)
                self._secure_store_files()
            self._schema_ready = True

    @staticmethod
    def _quoted_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    def _validate_identifier_migration_preflight(
        self, conn: sqlite3.Connection
    ) -> None:
        """Reject lossy identifiers and canonical-name collisions before DDL."""
        real_identifiers = conn.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM telegram_inbound_event "
            " WHERE typeof(bot_account_id)='real' OR typeof(update_id)='real') + "
            "(SELECT COUNT(*) FROM telegram_inbound_alias "
            " WHERE typeof(bot_account_id)='real')"
        ).fetchone()[0]
        if real_identifiers:
            raise RuntimeError(
                "Telegram inbound migration found lossy REAL identifiers; "
                "restore or repair the database before startup"
            )

        canonical_indexes = {
            "telegram_inbound_event_account_update": (
                "telegram_inbound_event",
                True,
                ("bot_account_id", "update_id"),
            ),
            "telegram_inbound_event_ready": (
                "telegram_inbound_event",
                False,
                ("bot_account_id", "work_state", "dispatch_state", "priority", "seq"),
            ),
            "telegram_inbound_alias_event": (
                "telegram_inbound_alias",
                False,
                ("event_id",),
            ),
        }
        for name, (expected_table, expected_unique, expected_columns) in (
            canonical_indexes.items()
        ):
            row = conn.execute(
                "SELECT tbl_name FROM sqlite_master WHERE type='index' AND name=?",
                (name,),
            ).fetchone()
            if row is None:
                continue
            table = str(row[0])
            index_entry = next(
                (
                    item
                    for item in conn.execute(
                        f"PRAGMA index_list({self._quoted_identifier(table)})"
                    ).fetchall()
                    if str(item[1]) == name
                ),
                None,
            )
            columns = tuple(
                str(item[2])
                for item in conn.execute(
                    f"PRAGMA index_info({self._quoted_identifier(name)})"
                ).fetchall()
            )
            if (
                table != expected_table
                or index_entry is None
                or bool(index_entry[2]) != expected_unique
                or bool(index_entry[4])
                or columns != expected_columns
            ):
                raise RuntimeError(
                    "Telegram inbound migration found conflicting canonical index "
                    f"{name}"
                )

    def _migrate_identifier_affinity(self, conn: sqlite3.Connection) -> None:
        """Rebuild legacy INTEGER-affinity identity columns as lossless TEXT."""
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._validate_identifier_migration_preflight(conn)
            sequence_row = conn.execute(
                "SELECT seq FROM sqlite_sequence "
                "WHERE name='telegram_inbound_event'"
            ).fetchone()
            legacy_sequence = int(sequence_row[0]) if sequence_row is not None else 0
            index_rows = conn.execute(
                "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' "
                "AND tbl_name IN ('telegram_inbound_event', 'telegram_inbound_alias') "
                "AND sql IS NOT NULL"
            ).fetchall()
            canonical_index_names = {
                "telegram_inbound_event_account_update",
                "telegram_inbound_event_ready",
                "telegram_inbound_alias_event",
            }
            preserved_indexes: list[tuple[str, str]] = []
            for raw_name, _raw_table, raw_sql in index_rows:
                name = str(raw_name)
                sql = str(raw_sql)
                if name not in canonical_index_names:
                    preserved_indexes.append((name, sql))
            for raw_name, _raw_table, _raw_sql in index_rows:
                name = str(raw_name)
                conn.execute(f"DROP INDEX {self._quoted_identifier(name)}")

            conn.execute(
                "ALTER TABLE telegram_inbound_alias "
                "RENAME TO telegram_inbound_alias_legacy"
            )
            conn.execute(
                "ALTER TABLE telegram_inbound_event "
                "RENAME TO telegram_inbound_event_legacy"
            )
            conn.execute(
                """CREATE TABLE telegram_inbound_event (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    bot_account_id TEXT NOT NULL,
                    update_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    update_kind TEXT NOT NULL,
                    chat_id TEXT,
                    message_id TEXT,
                    callback_query_id TEXT,
                    session_key TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    payload_json TEXT,
                    payload_sha256 TEXT,
                    work_state TEXT NOT NULL DEFAULT 'queued',
                    receipt_state TEXT NOT NULL DEFAULT 'not_required',
                    dispatch_state TEXT NOT NULL DEFAULT 'pending',
                    next_attempt_at REAL,
                    received_at REAL NOT NULL,
                    persisted_at REAL NOT NULL,
                    queued_at REAL,
                    lease_owner TEXT,
                    lease_epoch INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL,
                    leased_at REAL,
                    context_committed_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    replay_count INTEGER NOT NULL DEFAULT 0,
                    last_replay_at REAL,
                    last_error_class TEXT,
                    duplicate_count INTEGER NOT NULL DEFAULT 0,
                    identity_conflict_count INTEGER NOT NULL DEFAULT 0,
                    last_duplicate_at REAL,
                    consumed_at REAL,
                    terminal_at REAL,
                    receipt_attempted_at REAL,
                    receipt_confirmed_at REAL,
                    terminal_reason TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE telegram_inbound_alias (
                    bot_account_id TEXT NOT NULL,
                    alias_kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    alias_value TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    PRIMARY KEY (bot_account_id, alias_kind, scope, alias_value),
                    FOREIGN KEY (event_id) REFERENCES telegram_inbound_event(event_id)
                )"""
            )

            old_event_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(telegram_inbound_event_legacy)"
                ).fetchall()
            }
            new_event_columns = [
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(telegram_inbound_event)"
                ).fetchall()
                if str(row[1]) in old_event_columns
            ]
            event_targets = ", ".join(
                self._quoted_identifier(column) for column in new_event_columns
            )
            event_sources = ", ".join(
                f"CAST({self._quoted_identifier(column)} AS TEXT)"
                if column in {"bot_account_id", "update_id"}
                else self._quoted_identifier(column)
                for column in new_event_columns
            )
            conn.execute(
                f"INSERT INTO telegram_inbound_event ({event_targets}) "
                f"SELECT {event_sources} FROM telegram_inbound_event_legacy"
            )

            old_alias_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(telegram_inbound_alias_legacy)"
                ).fetchall()
            }
            new_alias_columns = [
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(telegram_inbound_alias)"
                ).fetchall()
                if str(row[1]) in old_alias_columns
            ]
            alias_targets = ", ".join(
                self._quoted_identifier(column) for column in new_alias_columns
            )
            alias_sources = ", ".join(
                f"CAST({self._quoted_identifier(column)} AS TEXT)"
                if column == "bot_account_id"
                else self._quoted_identifier(column)
                for column in new_alias_columns
            )
            conn.execute(
                f"INSERT INTO telegram_inbound_alias ({alias_targets}) "
                f"SELECT {alias_sources} FROM telegram_inbound_alias_legacy"
            )
            conn.execute("DROP TABLE telegram_inbound_alias_legacy")
            conn.execute("DROP TABLE telegram_inbound_event_legacy")
            copied_sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM telegram_inbound_event"
                ).fetchone()[0]
            )
            target_sequence = max(legacy_sequence, copied_sequence)
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name='telegram_inbound_event'"
            )
            if target_sequence:
                conn.execute(
                    "INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)",
                    ("telegram_inbound_event", target_sequence),
                )
            for _name, sql in preserved_indexes:
                conn.execute(sql)
            conn.execute(
                "CREATE UNIQUE INDEX "
                "telegram_inbound_event_account_update "
                "ON telegram_inbound_event(bot_account_id, update_id)"
            )
            conn.execute(
                "CREATE INDEX telegram_inbound_event_ready "
                "ON telegram_inbound_event(bot_account_id, work_state, "
                "dispatch_state, priority, seq)"
            )
            conn.execute(
                "CREATE INDEX telegram_inbound_alias_event "
                "ON telegram_inbound_alias(event_id)"
            )
            foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise RuntimeError(
                    "Telegram inbound identifier migration violated foreign keys"
                )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

    @staticmethod
    def _payload_bytes(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @staticmethod
    def _event_id(account: str, update_id: str) -> str:
        return f"telegram:{account}:{update_id}"


    @staticmethod
    def _row(row: Optional[sqlite3.Row], *, replayed: bool = False) -> Optional[InboundEvent]:
        if row is None:
            return None
        payload = None
        if row["payload_json"] is not None:
            try:
                value = json.loads(row["payload_json"])
                if isinstance(value, dict):
                    payload = value
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = None
        return InboundEvent(
            event_id=str(row["event_id"]),
            bot_account_id=int(row["bot_account_id"]),
            update_id=int(row["update_id"]),
            profile=str(row["profile"]),
            account_id=str(row["account_id"]),
            update_kind=str(row["update_kind"]),
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            callback_query_id=row["callback_query_id"],
            session_key=str(row["session_key"]),
            priority=int(row["priority"]),
            payload=payload,
            work_state=str(row["work_state"]),
            dispatch_state=str(row["dispatch_state"]),
            received_at=float(row["received_at"]),
            persisted_at=float(row["persisted_at"]),
            queued_at=(float(row["queued_at"]) if row["queued_at"] is not None else None),
            lease_owner=row["lease_owner"],
            lease_epoch=int(row["lease_epoch"]),
            lease_expires_at=(
                float(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            leased_at=(float(row["leased_at"]) if row["leased_at"] is not None else None),
            attempt_count=int(row["attempt_count"]),
            replay_count=int(row["replay_count"]),
            last_replay_at=(
                float(row["last_replay_at"])
                if row["last_replay_at"] is not None
                else None
            ),
            last_error_class=row["last_error_class"],
            duplicate_count=int(row["duplicate_count"]),
            identity_conflict_count=int(row["identity_conflict_count"]),
            last_duplicate_at=(
                float(row["last_duplicate_at"])
                if row["last_duplicate_at"] is not None
                else None
            ),
            consumed_at=(float(row["consumed_at"]) if row["consumed_at"] is not None else None),
            terminal_at=(float(row["terminal_at"]) if row["terminal_at"] is not None else None),
            terminal_reason=row["terminal_reason"],
            payload_sha256=row["payload_sha256"],
            replayed=replayed,
        )

    def persist(
        self,
        *,
        bot_account_id: Any,
        update_id: Any,
        decision: CaptureDecision,
        now: Optional[float] = None,
    ) -> PersistResult:
        """Persist one actionable update before the caller acknowledges it."""
        if not decision.actionable:
            raise ValueError("non-actionable updates are not durable work")
        account = canonical_bot_account_id(bot_account_id)
        update = _canonical_decimal(update_id, "update_id")
        when = time.time() if now is None else float(now)
        payload = decision.payload
        if not isinstance(payload, dict):
            raise ValueError("actionable Telegram updates require a dictionary payload")
        payload_bytes = self._payload_bytes(payload)
        payload_json = payload_bytes.decode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        event_id = self._event_id(account, update)

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM telegram_inbound_event "
                "WHERE bot_account_id=? AND update_id=?",
                (account, update),
            ).fetchone()
            if existing is not None:
                same_update = str(existing["update_id"]) == update
                conflict = bool(same_update and existing["payload_sha256"] != payload_hash)
                conn.execute(
                    "UPDATE telegram_inbound_event SET duplicate_count=duplicate_count+1, "
                    "identity_conflict_count=identity_conflict_count+?, "
                    "last_duplicate_at=? WHERE event_id=?",
                    (1 if conflict else 0, when, existing["event_id"]),
                )
                refreshed = conn.execute(
                    "SELECT * FROM telegram_inbound_event WHERE event_id=?",
                    (existing["event_id"],),
                ).fetchone()
                return PersistResult(
                    event_id=str(refreshed["event_id"]),
                    duplicate=True,
                    identity_conflict=conflict,
                    row=self._row(refreshed),
                )

            conn.execute(
                """INSERT INTO telegram_inbound_event(
                    event_id, bot_account_id, update_id, profile, account_id,
                    update_kind, chat_id, message_id, callback_query_id, session_key,
                    priority, payload_json, payload_sha256, work_state, receipt_state,
                    dispatch_state, received_at, persisted_at, queued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued',
                          'not_required', 'pending', ?, ?, ?)""",
                (
                    event_id,
                    account,
                    update,
                    str(decision.profile),
                    str(decision.account_id),
                    str(decision.update_kind),
                    None if decision.chat_id is None else str(decision.chat_id),
                    None if decision.message_id is None else str(decision.message_id),
                    None
                    if decision.callback_query_id is None
                    else str(decision.callback_query_id),
                    str(decision.session_key),
                    int(decision.priority),
                    payload_json,
                    payload_hash,
                    when,
                    when,
                    when,
                ),
            )
            inserted = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE event_id=?", (event_id,)
            ).fetchone()
            return PersistResult(
                event_id=event_id,
                duplicate=False,
                identity_conflict=False,
                row=self._row(inserted),
            )

    def get(self, event_id: str) -> Optional[InboundEvent]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._row(row)

    def event_id_for_update(self, bot_account_id: Any, update_id: Any) -> Optional[str]:
        account = canonical_bot_account_id(bot_account_id)
        update = _canonical_decimal(update_id, "update_id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT event_id FROM telegram_inbound_event "
                "WHERE bot_account_id=? AND update_id=?",
                (account, update),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def _account_filter(self, bot_account_id: Optional[Any]) -> tuple[str, tuple[Any, ...]]:
        if bot_account_id is None:
            return "", ()
        return " AND bot_account_id=?", (canonical_bot_account_id(bot_account_id),)

    def reclaim_process_leases(
        self,
        *,
        bot_account_id: Any,
        current_owner: str,
        now: Optional[float] = None,
        exclude_event_ids: tuple[str, ...] = (),
    ) -> int:
        """Requeue abandoned leases while preserving known active local work."""
        account = canonical_bot_account_id(bot_account_id)
        when = time.time() if now is None else float(now)
        excluded = tuple(str(event_id) for event_id in exclude_event_ids)
        exclusion = (
            " AND event_id NOT IN (" + ",".join("?" for _ in excluded) + ")"
            if excluded
            else ""
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return int(
                conn.execute(
                    """UPDATE telegram_inbound_event SET
                        work_state=CASE
                            WHEN terminal_reason='control_effect_started'
                                 OR attempt_count>=? THEN 'dead_letter'
                            ELSE 'queued'
                        END,
                        lease_owner=NULL, lease_expires_at=NULL,
                        next_attempt_at=CASE
                            WHEN terminal_reason='control_effect_started'
                                 OR attempt_count>=? THEN NULL
                            ELSE ?
                        END,
                        dispatch_state='pending',
                        replay_count=replay_count+1, last_replay_at=?,
                        terminal_at=CASE
                            WHEN terminal_reason='control_effect_started'
                                 OR attempt_count>=? THEN COALESCE(terminal_at, ?)
                            ELSE terminal_at
                        END,
                        terminal_reason=CASE
                            WHEN terminal_reason='control_effect_started'
                                THEN 'control_effect_failed'
                            WHEN attempt_count>=? THEN 'retry_budget_exhausted'
                            ELSE terminal_reason
                        END
                       WHERE bot_account_id=? AND work_state='leased'
                         AND (lease_owner<>? OR lease_expires_at<=?)"""
                    + exclusion,
                    (
                        MAX_ATTEMPTS,
                        MAX_ATTEMPTS,
                        when,
                        when,
                        MAX_ATTEMPTS,
                        when,
                        MAX_ATTEMPTS,
                        account,
                        str(current_owner),
                        when,
                        *excluded,
                    ),
                ).rowcount
            )

    def recover_admitted_dispatches(
        self, *, bot_account_id: Any, now: Optional[float] = None
    ) -> int:
        """Make pre-acknowledged projections eligible for replay for one bot."""
        account = canonical_bot_account_id(bot_account_id)
        when = time.time() if now is None else float(now)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return int(
                conn.execute(
                    """UPDATE telegram_inbound_event SET dispatch_state='pending',
                        replay_count=replay_count+1, last_replay_at=?
                       WHERE bot_account_id=? AND work_state='queued'
                         AND dispatch_state='admitted'""",
                    (when, account),
                ).rowcount
            )

    def handoff_owner_leases(
        self,
        *,
        bot_account_id: Any,
        old_owner: str,
        new_owner: str,
        live_event_ids: set[str],
        now: Optional[float] = None,
    ) -> dict[str, int]:
        """Transfer live handler claims and recover abandoned claims."""
        account = canonical_bot_account_id(bot_account_id)
        old = str(old_owner)
        new = str(new_owner)
        when = time.time() if now is None else float(now)
        counts = {"requeued": 0, "transferred": 0, "quarantined": 0}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT event_id, terminal_reason, attempt_count
                   FROM telegram_inbound_event
                   WHERE bot_account_id=? AND work_state='leased' AND lease_owner=?""",
                (account, old),
            ).fetchall()
            for row in rows:
                event_id = str(row["event_id"])
                # A one-shot control effect is ambiguous once its external
                # action may have started.  Quarantine it even if its handler
                # is still live; transferring it could repeat the side effect.
                if row["terminal_reason"] == "control_effect_started":
                    changed = conn.execute(
                        """UPDATE telegram_inbound_event SET work_state='dead_letter',
                              lease_owner=NULL, lease_expires_at=NULL,
                              next_attempt_at=NULL, terminal_at=?, terminal_reason=?,
                              last_error_class=?
                           WHERE event_id=? AND bot_account_id=?
                             AND work_state='leased' AND lease_owner=?""",
                        (
                            when,
                            "ambiguous_reconnect_after_control_start",
                            "reconnect_after_control_start",
                            event_id,
                            account,
                            old,
                        ),
                    ).rowcount
                    if changed == 1:
                        counts["quarantined"] += 1
                    continue
                if event_id in live_event_ids:
                    changed = conn.execute(
                        """UPDATE telegram_inbound_event SET lease_owner=?
                           WHERE event_id=? AND bot_account_id=?
                             AND work_state='leased' AND lease_owner=?""",
                        (new, event_id, account, old),
                    ).rowcount
                    if changed == 1:
                        counts["transferred"] += 1
                    continue
                if int(row["attempt_count"]) >= MAX_ATTEMPTS:
                    changed = conn.execute(
                        """UPDATE telegram_inbound_event SET work_state='dead_letter',
                              lease_owner=NULL, lease_expires_at=NULL,
                              next_attempt_at=NULL, terminal_at=?,
                              terminal_reason=?, last_error_class=?
                           WHERE event_id=? AND bot_account_id=?
                             AND work_state='leased' AND lease_owner=?""",
                        (
                            when,
                            "retry_budget_exhausted",
                            "retry_budget_exhausted",
                            event_id,
                            account,
                            old,
                        ),
                    ).rowcount
                    if changed == 1:
                        counts["quarantined"] += 1
                    continue
                changed = conn.execute(
                    """UPDATE telegram_inbound_event SET work_state='queued',
                          lease_owner=NULL, lease_expires_at=NULL,
                          dispatch_state='pending', replay_count=replay_count+1,
                          last_replay_at=?
                       WHERE event_id=? AND bot_account_id=?
                         AND work_state='leased' AND lease_owner=?""",
                    (when, event_id, account, old),
                ).rowcount
                if changed == 1:
                    counts["requeued"] += 1
        return counts

    def _ready_where(self, *, include_expired_lease: bool = True) -> str:
        if include_expired_lease:
            return "((work_state='queued' AND " \
                "(next_attempt_at IS NULL OR next_attempt_at<=?)) OR " \
                "(work_state='leased' AND lease_expires_at<=?))"
        return "work_state='queued' AND (next_attempt_at IS NULL OR next_attempt_at<=?)"

    def eligible(
        self,
        *,
        bot_account_id: Optional[Any] = None,
        now: Optional[float] = None,
        limit: int = 32,
    ) -> list[InboundEvent]:
        when = time.time() if now is None else float(now)
        bounded = max(1, int(limit))
        clause, params = self._account_filter(bot_account_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE "
                + self._ready_where()
                + clause
                + " ORDER BY priority,seq LIMIT ?",
                (when, when, *params, bounded),
            ).fetchall()
        return [self._row(row, replayed=row["work_state"] == "leased") for row in rows]

    def pending_dispatch(
        self,
        *,
        bot_account_id: Optional[Any] = None,
        now: Optional[float] = None,
        limit: int = 32,
    ) -> list[InboundEvent]:
        when = time.time() if now is None else float(now)
        bounded = max(1, int(limit))
        clause, params = self._account_filter(bot_account_id)
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM telegram_inbound_event
                   WHERE dispatch_state='pending' AND work_state='queued'
                     AND (next_attempt_at IS NULL OR next_attempt_at<=?)"""
                + clause
                + " ORDER BY priority,seq LIMIT ?",
                (when, *params, bounded),
            ).fetchall()
        return [self._row(row) for row in rows]

    def next_pending_dispatch_at(
        self, *, bot_account_id: Optional[Any] = None
    ) -> Optional[float]:
        """Return the earliest delayed pending row for an account."""
        clause, params = self._account_filter(bot_account_id)
        with self._connect() as conn:
            row = conn.execute(
                """SELECT MIN(next_attempt_at) FROM telegram_inbound_event
                   WHERE dispatch_state='pending' AND work_state='queued'
                     AND next_attempt_at IS NOT NULL"""
                + clause,
                params,
            ).fetchone()
        return float(row[0]) if row is not None and row[0] is not None else None

    def lease_next(
        self,
        *,
        owner: str,
        bot_account_id: Optional[Any] = None,
        now: Optional[float] = None,
        lease_seconds: float = 30.0,
    ) -> Optional[InboundEvent]:
        when = time.time() if now is None else float(now)
        clause, params = self._account_filter(bot_account_id)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE "
                + self._ready_where()
                + clause
                + " ORDER BY priority,seq LIMIT 1",
                (when, when, *params),
            ).fetchone()
            if row is None:
                return None
            replayed = row["work_state"] == "leased"
            old_epoch = int(row["lease_epoch"])
            epoch = old_epoch + 1
            changed = conn.execute(
                """UPDATE telegram_inbound_event SET work_state='leased',
                    lease_owner=?, lease_epoch=?, lease_expires_at=?, leased_at=?,
                    attempt_count=attempt_count+1, replay_count=replay_count+?,
                    last_replay_at=CASE WHEN ? THEN ? ELSE last_replay_at END
                   WHERE event_id=? AND lease_epoch=? AND
                         (work_state='queued' OR
                          (work_state='leased' AND lease_expires_at<=?))""",
                (
                    str(owner),
                    epoch,
                    when + float(lease_seconds),
                    when,
                    1 if replayed else 0,
                    replayed,
                    when,
                    row["event_id"],
                    old_epoch,
                    when,
                ),
            ).rowcount
            if changed != 1:
                return None
            leased = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE event_id=?",
                (row["event_id"],),
            ).fetchone()
        return self._row(leased, replayed=replayed)

    def lease_event(
        self,
        event_id: str,
        *,
        owner: str,
        now: Optional[float] = None,
        lease_seconds: float = 86400.0,
    ) -> Optional[InboundEvent]:
        """Lease one row with an epoch-fenced compare-and-set."""
        when = time.time() if now is None else float(now)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None or row["work_state"] in _TERMINAL_STATES:
                return None
            if row["work_state"] == "leased" and row["lease_expires_at"] > when:
                if row["lease_owner"] == str(owner):
                    return self._row(row)
                return None
            replayed = row["work_state"] == "leased"
            old_epoch = int(row["lease_epoch"])
            changed = conn.execute(
                """UPDATE telegram_inbound_event SET work_state='leased',
                    lease_owner=?, lease_epoch=?, lease_expires_at=?, leased_at=?,
                    attempt_count=attempt_count+1, replay_count=replay_count+?,
                    last_replay_at=CASE WHEN ? THEN ? ELSE last_replay_at END
                   WHERE event_id=? AND lease_epoch=? AND
                         (work_state='queued' OR
                          (work_state='leased' AND lease_expires_at<=?))""",
                (
                    str(owner),
                    old_epoch + 1,
                    when + float(lease_seconds),
                    when,
                    1 if replayed else 0,
                    replayed,
                    when,
                    event_id,
                    old_epoch,
                    when,
                ),
            ).rowcount
            if changed != 1:
                return None
            leased = conn.execute(
                "SELECT * FROM telegram_inbound_event WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._row(leased, replayed=replayed)

    def mark_consumed(
        self,
        event_id: str,
        *,
        owner: str,
        lease_epoch: int,
        now: Optional[float] = None,
    ) -> bool:
        when = time.time() if now is None else float(now)
        with self._connect() as conn:
            return (
                conn.execute(
                    """UPDATE telegram_inbound_event SET work_state='consumed',
                        lease_owner=NULL, lease_expires_at=NULL, consumed_at=?,
                        terminal_at=?
                       WHERE event_id=? AND work_state='leased' AND lease_owner=?
                         AND lease_epoch=?""",
                    (when, when, event_id, str(owner), int(lease_epoch)),
                ).rowcount
                == 1
            )

    def requeue(
        self,
        event_id: str,
        *,
        owner: str,
        lease_epoch: int,
        now: Optional[float] = None,
        error_class: str = "retry",
        delay: float = 0.0,
        preserve_retry_budget: bool = False,
    ) -> bool:
        when = time.time() if now is None else float(now)
        with self._connect() as conn:
            if preserve_retry_budget:
                # A defer means the handler did not enter owned processing.
                # Undo this lease's attempt and any provisional control marker
                # instead of treating backpressure as a failed side effect.
                return (
                    conn.execute(
                        """UPDATE telegram_inbound_event SET
                            attempt_count=CASE
                                WHEN attempt_count>0 THEN attempt_count-1
                                ELSE 0
                            END,
                            work_state='queued',
                            lease_owner=NULL, lease_expires_at=NULL,
                            dispatch_state='pending', next_attempt_at=?,
                            last_error_class=?,
                            terminal_at=CASE
                                WHEN terminal_reason='control_effect_started' THEN NULL
                                ELSE terminal_at
                            END,
                            terminal_reason=CASE
                                WHEN terminal_reason='control_effect_started' THEN NULL
                                ELSE terminal_reason
                            END
                           WHERE event_id=? AND work_state='leased' AND lease_owner=?
                             AND lease_epoch=?""",
                        (
                            when + max(0.0, float(delay)),
                            str(error_class)[:80],
                            event_id,
                            str(owner),
                            int(lease_epoch),
                        ),
                    ).rowcount
                    == 1
                )

            retry_enabled = 1
            return (
                conn.execute(
                    """UPDATE telegram_inbound_event SET
                        attempt_count=CASE
                            WHEN terminal_reason='control_effect_started' THEN attempt_count
                            WHEN ?=0 AND attempt_count>0 THEN attempt_count-1
                            ELSE attempt_count
                        END,
                        work_state=CASE
                            WHEN terminal_reason='control_effect_started'
                                 OR (attempt_count>=? AND ?=1) THEN 'dead_letter'
                            ELSE 'queued'
                        END,
                        lease_owner=NULL, lease_expires_at=NULL,
                        dispatch_state='pending',
                        next_attempt_at=CASE
                            WHEN terminal_reason='control_effect_started'
                                 OR (attempt_count>=? AND ?=1) THEN NULL
                            ELSE ?
                        END,
                        last_error_class=?,
                        terminal_at=CASE
                            WHEN terminal_reason='control_effect_started'
                                 OR (attempt_count>=? AND ?=1) THEN COALESCE(terminal_at, ?)
                            ELSE terminal_at
                        END,
                        terminal_reason=CASE
                            WHEN terminal_reason='control_effect_started'
                                THEN 'control_effect_failed'
                            WHEN attempt_count>=? AND ?=1 THEN 'retry_budget_exhausted'
                            ELSE terminal_reason
                        END
                       WHERE event_id=? AND work_state='leased' AND lease_owner=?
                         AND lease_epoch=?""",
                    (
                        retry_enabled,
                        MAX_ATTEMPTS,
                        retry_enabled,
                        MAX_ATTEMPTS,
                        retry_enabled,
                        when + max(0.0, float(delay)),
                        str(error_class)[:80],
                        MAX_ATTEMPTS,
                        retry_enabled,
                        when,
                        MAX_ATTEMPTS,
                        retry_enabled,
                        event_id,
                        str(owner),
                        int(lease_epoch),
                    ),
                ).rowcount
                == 1
            )

    def mark_control_started(
        self, event_id: str, *, owner: str, lease_epoch: int
    ) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    """UPDATE telegram_inbound_event SET terminal_reason='control_effect_started'
                       WHERE event_id=? AND update_kind='callback_query'
                         AND work_state='leased' AND lease_owner=? AND lease_epoch=?""",
                    (event_id, str(owner), int(lease_epoch)),
                ).rowcount
                == 1
            )

    def mark_dispatch_admitted(self, event_id: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    """UPDATE telegram_inbound_event SET dispatch_state='admitted'
                       WHERE event_id=? AND dispatch_state='pending'
                         AND work_state IN ('queued','leased')""",
                    (event_id,),
                ).rowcount
                == 1
            )

    def reset_dispatch_pending(self, event_id: str) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    """UPDATE telegram_inbound_event SET dispatch_state='pending'
                       WHERE event_id=? AND dispatch_state='admitted'
                         AND work_state IN ('queued','leased')""",
                    (event_id,),
                ).rowcount
                == 1
            )

    def reconcile_consumed(
        self,
        *,
        bot_account_id: Any,
        committed_event_ids: set[str],
        now: Optional[float] = None,
    ) -> int:
        """Consume only supplied event ids belonging to the supplied bot."""
        if not committed_event_ids:
            return 0
        account = canonical_bot_account_id(bot_account_id)
        when = time.time() if now is None else float(now)
        event_ids = tuple(sorted(str(value) for value in committed_event_ids))
        placeholders = ",".join("?" for _ in event_ids)
        with self._connect() as conn:
            return int(
                conn.execute(
                    """UPDATE telegram_inbound_event SET work_state='consumed',
                        lease_owner=NULL, lease_expires_at=NULL, consumed_at=?,
                        terminal_at=?
                       WHERE bot_account_id=? AND work_state IN ('queued','leased')
                         AND event_id IN ("""
                    + placeholders
                    + ")",
                    (when, when, account, *event_ids),
                ).rowcount
            )

    def health_snapshot(
        self, *, now: Optional[float] = None, stale_after: float = 60.0
    ) -> dict[str, Any]:
        when = time.time() if now is None else float(now)
        with self._connect() as conn:
            counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT work_state, COUNT(*) FROM telegram_inbound_event GROUP BY work_state"
                )
            }
            aggregate = conn.execute(
                """SELECT COALESCE(SUM(duplicate_count),0),
                          COALESCE(SUM(identity_conflict_count),0),
                          COALESCE(SUM(replay_count),0),
                          MIN(CASE WHEN work_state='queued' THEN queued_at END),
                          MAX(persisted_at), MAX(consumed_at)
                     FROM telegram_inbound_event"""
            ).fetchone()
        oldest = max(0.0, when - float(aggregate[3])) if aggregate[3] is not None else 0.0
        reasons: list[str] = []
        if aggregate[1]:
            reasons.append("identity_conflict")
        if oldest > float(stale_after):
            reasons.append("stale_user_work")
        return {
            "status": "degraded" if reasons else "ready",
            "schema_version": SCHEMA_VERSION,
            "queued": counts.get("queued", 0),
            "leased": counts.get("leased", 0),
            "consumed": counts.get("consumed", 0),
            "dead_letter": counts.get("dead_letter", 0),
            "duplicates": int(aggregate[0]),
            "identity_conflicts": int(aggregate[1]),
            "replayed": int(aggregate[2]),
            "oldest_queued_age_seconds": oldest,
            "last_persisted_at": aggregate[4],
            "last_consumed_at": aggregate[5],
            "reasons": reasons,
        }

    @classmethod
    def health_snapshot_readonly(
        cls, path: Path, *, now: Optional[float] = None, stale_after: float = 60.0
    ) -> dict[str, Any]:
        """Read an existing store without creating or mutating it."""
        store_path = Path(path)
        if not store_path.exists():
            return {"status": "unknown", "reasons": ["missing_store"]}
        when = time.time() if now is None else float(now)
        conn = sqlite3.connect(f"file:{store_path.as_posix()}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT work_state, COUNT(*) FROM telegram_inbound_event GROUP BY work_state"
                )
            }
            aggregate = conn.execute(
                """SELECT COALESCE(SUM(duplicate_count),0),
                          COALESCE(SUM(identity_conflict_count),0),
                          COALESCE(SUM(replay_count),0),
                          MIN(CASE WHEN work_state='queued' THEN queued_at END),
                          MAX(persisted_at), MAX(consumed_at)
                     FROM telegram_inbound_event"""
            ).fetchone()
        except sqlite3.Error:
            return {"status": "unknown", "reasons": ["invalid_store"]}
        finally:
            conn.close()
        oldest = max(0.0, when - float(aggregate[3])) if aggregate[3] is not None else 0.0
        reasons = ["identity_conflict"] if aggregate[1] else []
        if oldest > float(stale_after):
            reasons.append("stale_user_work")
        return {
            "status": "degraded" if reasons else "ready",
            "schema_version": SCHEMA_VERSION,
            "queued": counts.get("queued", 0),
            "leased": counts.get("leased", 0),
            "consumed": counts.get("consumed", 0),
            "dead_letter": counts.get("dead_letter", 0),
            "duplicates": int(aggregate[0]),
            "identity_conflicts": int(aggregate[1]),
            "replayed": int(aggregate[2]),
            "oldest_queued_age_seconds": oldest,
            "last_persisted_at": aggregate[4],
            "last_consumed_at": aggregate[5],
            "reasons": reasons,
        }


class DurableTelegramUpdateQueue(asyncio.Queue):
    """Queue that durably captures actionable PTB updates before returning."""

    def __init__(
        self,
        *,
        store: TelegramInboundStore,
        bot_account_id: Any,
        classifier: Callable[[Any], CaptureDecision],
        after_commit: Optional[
            Callable[[Any, PersistResult], Awaitable[None] | None]
        ] = None,
        lease_owner: Optional[str] = None,
        active_limit: int = 32,
        item_factory: Optional[Callable[[Any], Any]] = None,
        maxsize: Optional[int] = None,
    ):
        if maxsize is not None and int(maxsize) > 0:
            active_limit = int(maxsize)
        self.active_limit = max(1, int(active_limit))
        # The physical PTB queue must remain unbounded: rejected updates and
        # shutdown sentinels retain ordinary asyncio.Queue semantics even when
        # the durable projection window is full. ``active_limit`` below bounds
        # only durable rows projected into this queue.
        super().__init__(maxsize=0)
        self.store = store
        self.bot_account_id = int(canonical_bot_account_id(bot_account_id))
        self.classifier = classifier
        self.after_commit = after_commit
        self.item_factory = item_factory or (lambda payload: payload)
        self.lease_owner = lease_owner or f"telegram:{os.getpid()}:{uuid.uuid4().hex}"
        self._admission_lock = asyncio.Lock()
        # Durable admission must not share asyncio's process-wide default
        # executor. A saturated unrelated worker pool otherwise blocks PTB's
        # update_queue.put() acknowledgment boundary before SQLite is reached.
        self._store_executor_key = (
            f"{self.store.lifecycle_path}:{canonical_bot_account_id(self.bot_account_id)}"
        )
        self._store_executor = _acquire_store_executor(
            self._store_executor_key, self.bot_account_id
        )
        self._persist_retry_initial_seconds = _INGRESS_RETRY_INITIAL_SECONDS
        self._persist_retry_max_seconds = _INGRESS_RETRY_MAX_SECONDS
        self._persist_stall_log_seconds = _INGRESS_STALL_LOG_SECONDS
        self._ingress_pause_lock = threading.Lock()
        self._ingress_paused_at: Optional[float] = None
        self._ingress_pause_stage: Optional[str] = None
        self._ingress_pause_error_class: Optional[str] = None
        self._ingress_pause_attempt = 0
        self._ingress_pause_last_log_at: Optional[float] = None
        self._ingress_tasks: set[asyncio.Task[Any]] = set()
        # ``put`` is PTB's acknowledgement boundary.  A handoff must close
        # admission before it retires the predecessor executor, then wait for
        # every already-admitted put (including cancellation recovery) to stop
        # submitting SQLite work.
        self._ingress_handoff_lock = asyncio.Lock()
        self._ingress_owner_tasks: set[asyncio.Task[Any]] = set()
        self._ingress_closed = False
        self._projection_retry_task: Optional[asyncio.Task[Any]] = None
        self._projection_suspended = False
        self._store_executor_shutdown = False
        self._claim_handoff_lock = threading.RLock()
        self._queued_event_ids: set[str] = set()
        self._claiming_event_ids: set[str] = set()
        self._claiming_done: dict[str, asyncio.Event] = {}
        self._event_id_by_update_id: dict[int, str] = {}
        self._handed_off_update_ids: set[int] = set()
        self._claim_by_update_id: dict[int, InboundEvent] = {}
        # A deferred cleanup can fail after the SQLite operation starts. Keep
        # only an exact still-leased claim available for a later cleanup retry;
        # it is intentionally not exposed as a handler claim or transferred
        # during queue handoff.
        self._retryable_claim_by_update_id: dict[int, InboundEvent] = {}
        self._claim_transition_tasks: dict[int, asyncio.Task[Any]] = {}
        self._handler_event_ids: set[str] = set()
        # Durable events buffered by an adapter (text/media batching or the
        # disconnect hold queue) are not active handler work.  Keep that state
        # separate so a fresh adapter can requeue them instead of transferring
        # a lease whose in-memory MessageEvent is stranded on the old adapter.
        self._buffered_event_ids: set[str] = set()
        # A handoff removes claims from this queue.  A handler can still enter
        # the wrapper after that removal, so retain a one-shot fence for each
        # published claim until the late wrapper observes it.
        self._fenced_handler_update_ids: set[int] = set()
        self._handoff_target: Optional[DurableTelegramUpdateQueue] = None
        self._lifecycle_retired = False
        self._handoff_in_progress = False
        self._handoff_complete = asyncio.Event()
        self._handoff_complete.set()
        self._lifecycle_tasks: set[asyncio.Task[Any]] = set()
        self._handoff_receiver_ref: Optional[weakref.ReferenceType[Any]] = None
        try:
            self._scheduler_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self._scheduler_loop = None
        self._due_wakeup_handle: Optional[asyncio.TimerHandle] = None
        self._due_wakeup_at: Optional[float] = None
        self._due_wakeup_task: Optional[asyncio.Task[Any]] = None
        self._prehandler_lease_seconds = 30.0
        self._pending_dispatch_resets: set[str] = set()

    @property
    def ingress_paused(self) -> bool:
        """Return whether a pre-commit ingress operation is backpressured."""
        with self._ingress_pause_lock:
            return self._ingress_paused_at is not None

    def ingress_pause_snapshot(self) -> dict[str, Any]:
        """Return payload-free state for watchdog and operator diagnostics."""
        now = time.monotonic()
        with self._ingress_pause_lock:
            paused_at = self._ingress_paused_at
            return {
                "paused": paused_at is not None,
                "stage": self._ingress_pause_stage,
                "error_class": self._ingress_pause_error_class,
                "attempt": self._ingress_pause_attempt,
                "paused_for_seconds": (
                    max(0.0, now - paused_at) if paused_at is not None else 0.0
                ),
            }

    def _note_ingress_pause(
        self, stage: str, error: BaseException, attempt: int
    ) -> None:
        now = time.monotonic()
        error_class = type(error).__name__
        with self._ingress_pause_lock:
            if self._ingress_paused_at is None:
                self._ingress_paused_at = now
            self._ingress_pause_stage = stage
            self._ingress_pause_error_class = error_class
            self._ingress_pause_attempt = attempt
            last_log = self._ingress_pause_last_log_at
            should_log = (
                last_log is None
                or now - last_log >= max(0.01, self._persist_stall_log_seconds)
            )
            if should_log:
                self._ingress_pause_last_log_at = now
        if should_log:
            logger.warning(
                "Telegram durable ingress paused before polling acknowledgment: "
                "bot_account_id=%s stage=%s attempt=%d error=%s",
                self.bot_account_id,
                stage,
                attempt,
                error_class,
            )

    def _clear_ingress_pause(self) -> None:
        now = time.monotonic()
        with self._ingress_pause_lock:
            paused_at = self._ingress_paused_at
            if paused_at is None:
                return
            duration = max(0.0, now - paused_at)
            attempts = self._ingress_pause_attempt
            self._ingress_paused_at = None
            self._ingress_pause_stage = None
            self._ingress_pause_error_class = None
            self._ingress_pause_attempt = 0
            self._ingress_pause_last_log_at = None
        logger.info(
            "Telegram durable ingress resumed: bot_account_id=%s "
            "paused_for=%.3fs attempts=%d",
            self.bot_account_id,
            duration,
            attempts,
        )

    async def _run_store(self, function: Callable[..., Any], /, *args, **kwargs) -> Any:
        """Run one SQLite operation outside the process-wide executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._store_executor, partial(function, *args, **kwargs)
        )

    def _discard_ingress_task(self, task: asyncio.Task[Any]) -> None:
        self._ingress_tasks.discard(task)
        try:
            task.exception()
        except BaseException:
            pass

    def _track_ingress_task(self, task: asyncio.Task[Any]) -> None:
        self._ingress_tasks.add(task)
        task.add_done_callback(self._discard_ingress_task)

    async def _enter_ingress(self) -> Optional["DurableTelegramUpdateQueue"]:
        """Either reserve this queue's ingress epoch or return its successor."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Telegram ingress requires an asyncio task")
        while True:
            async with self._ingress_handoff_lock:
                target = self._handoff_target
                if target is None:
                    if self._ingress_closed:
                        raise RuntimeError("Telegram durable ingress queue is closed")
                    self._ingress_owner_tasks.add(task)
                    return None
            if target is not None:
                return target
            await self._wait_for_claim_event(self._handoff_complete)

    def _leave_ingress(self) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._ingress_owner_tasks.discard(task)

    async def _wait_for_ingress_tasks(self) -> None:
        """Drain predecessor puts before its dedicated executor is retired."""
        while True:
            tasks = tuple(
                task
                for task in self._ingress_owner_tasks | self._ingress_tasks
                if task is not asyncio.current_task() and not task.done()
            )
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _classify_with_retry(self, item: Any, raw: Any) -> CaptureDecision:
        delay = max(0.0, float(self._persist_retry_initial_seconds))
        attempt = 0
        while True:
            attempt += 1
            try:
                decision = self.classifier(item)
                if not isinstance(decision, CaptureDecision):
                    raise TypeError("Telegram classifier must return CaptureDecision")
                if decision.actionable and decision.payload is None:
                    if not isinstance(raw, dict):
                        raise ValueError(
                            "actionable Telegram updates require serializable payload"
                        )
                    decision.payload = raw
            except Exception as exc:
                self._note_ingress_pause("classify", exc, attempt)
                await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2.0, self._persist_retry_initial_seconds),
                    self._persist_retry_max_seconds,
                )
                continue
            if not decision.actionable:
                self._clear_ingress_pause()
            return decision

    async def _persist_with_retry(
        self, *, update_id: int, decision: CaptureDecision
    ) -> PersistResult:
        delay = max(0.0, float(self._persist_retry_initial_seconds))
        stall_log_seconds = max(0.01, float(self._persist_stall_log_seconds))
        attempt = 0
        while True:
            attempt += 1
            operation = asyncio.create_task(
                self._run_store(
                    self.store.persist,
                    bot_account_id=self.bot_account_id,
                    update_id=update_id,
                    decision=decision,
                )
            )
            try:
                while True:
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(operation), timeout=stall_log_seconds
                        )
                        break
                    except asyncio.TimeoutError as exc:
                        # Do not submit a second write while the first thread may
                        # still commit. Keep the PTB offset fail-closed and expose
                        # the intentional pause to the polling watchdog instead.
                        self._note_ingress_pause("persist_wait", exc, attempt)
            except asyncio.CancelledError:
                operation.cancel()
                raise
            except Exception as exc:
                self._note_ingress_pause("persist", exc, attempt)
                await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2.0, self._persist_retry_initial_seconds),
                    self._persist_retry_max_seconds,
                )
                continue
            self._clear_ingress_pause()
            return result

    def _schedule_projection_retry(self) -> None:
        if self._projection_suspended or self._lifecycle_retired:
            return
        current = self._projection_retry_task
        if current is not None and not current.done():
            return

        async def retry() -> None:
            delay = max(0.0, float(self._persist_retry_initial_seconds))
            while True:
                try:
                    await self.wake_scheduler()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Telegram durable projection retry: bot_account_id=%s "
                        "error=%s",
                        self.bot_account_id,
                        type(exc).__name__,
                    )
                else:
                    if not self._pending_dispatch_resets:
                        return
                await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2.0, self._persist_retry_initial_seconds),
                    self._persist_retry_max_seconds,
                )

        self._projection_retry_task = asyncio.create_task(retry())
        self._track_ingress_task(self._projection_retry_task)

    @staticmethod
    def _raw_payload(item: Any) -> Any:
        if isinstance(item, dict):
            return item
        converter = getattr(item, "to_dict", None)
        if callable(converter):
            payload = converter()
            if isinstance(payload, dict):
                return payload
        return None

    @classmethod
    def _update_id(cls, item: Any, raw: Any = None) -> Optional[int]:
        if raw is None:
            raw = cls._raw_payload(item)
        value = raw.get("update_id") if isinstance(raw, dict) else getattr(item, "update_id", None)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return None
        try:
            return int(_canonical_decimal(value, "update_id"))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_update(cls, item: Any, raw: Any = None) -> bool:
        return cls._update_id(item, raw) is not None

    def claim_for_update(self, update_id: Optional[int]) -> Optional[InboundEvent]:
        if update_id is None:
            return None
        with self._claim_handoff_lock:
            return self._claim_by_update_id.get(int(update_id))

    def handler_claimed(self, update_id: Optional[int]) -> bool:
        if update_id is None:
            return False
        with self._claim_handoff_lock:
            claim = self._claim_by_update_id.get(int(update_id))
            return claim is not None and claim.event_id in self._handler_event_ids

    def mark_handler_claim(self, update_id: Optional[int]) -> Optional[InboundEvent]:
        with self._claim_handoff_lock:
            claim = self._claim_by_update_id.get(int(update_id)) if update_id is not None else None
            if claim is not None:
                self._handler_event_ids.add(claim.event_id)
            return claim

    def mark_event_buffered(
        self, update_ids: tuple[int, ...] | list[int] | set[int]
    ) -> None:
        """Mark durable claims whose MessageEvents await delayed dispatch."""
        with self._claim_handoff_lock:
            for update_id in update_ids:
                normalized_update_id = int(update_id)
                claim = self._claim_by_update_id.get(normalized_update_id)
                if claim is None:
                    claim = self._retryable_claim_by_update_id.get(normalized_update_id)
                if claim is not None:
                    self._buffered_event_ids.add(claim.event_id)

    def mark_event_active(
        self, update_ids: tuple[int, ...] | list[int] | set[int]
    ) -> bool:
        """Mark durable claims that have entered actual dispatch."""
        activated = False
        with self._claim_handoff_lock:
            for update_id in update_ids:
                normalized_update_id = int(update_id)
                claim = self._claim_by_update_id.get(normalized_update_id)
                if claim is None:
                    claim = self._retryable_claim_by_update_id.pop(
                        normalized_update_id, None
                    )
                    if claim is not None:
                        self._claim_by_update_id[normalized_update_id] = claim
                if claim is not None:
                    self._buffered_event_ids.discard(claim.event_id)
                    self._handler_event_ids.add(claim.event_id)
                    activated = True
        return activated

    def handler_claim_fenced(self, update_id: Optional[int]) -> bool:
        """Consume the fence for a handler that lost queue ownership."""
        if update_id is None:
            return False
        with self._claim_handoff_lock:
            normalized_update_id = int(update_id)
            if normalized_update_id not in self._fenced_handler_update_ids:
                return False
            self._fenced_handler_update_ids.remove(normalized_update_id)
            return True

    def clear_handler_claim(self, event_id: str) -> None:
        with self._claim_handoff_lock:
            self._handler_event_ids.discard(event_id)

    def run_claim_operation(
        self, claim: InboundEvent, operation: Callable[[InboundEvent], Any]
    ) -> Any:
        with self._claim_handoff_lock:
            return operation(claim)

    def _begin_claiming_locked(self, event_id: str) -> Optional[asyncio.Event]:
        """Register one SQLite claim operation under the handoff lock."""
        if event_id in self._claiming_event_ids:
            return None
        done = asyncio.Event()
        self._claiming_event_ids.add(event_id)
        self._claiming_done[event_id] = done
        return done

    def _finish_claiming(self, event_id: str) -> None:
        with self._claim_handoff_lock:
            self._claiming_event_ids.discard(event_id)
            done = self._claiming_done.pop(event_id, None)
            if done is not None:
                done.set()

    def _discard_claiming_projection(self, event_id: str) -> None:
        """Release both the claim fence and physical queue task debt."""
        try:
            self._finish_claiming(event_id)
        finally:
            super().task_done()

    def _discard_lifecycle_task(self, task: asyncio.Task[Any]) -> None:
        with self._claim_handoff_lock:
            self._lifecycle_tasks.discard(task)
        # A task can finish between the handoff barrier's snapshots. Retrieve
        # its exception here so an exceptional cleanup cannot become an
        # unobserved task warning; the adapter's callback records the failure.
        try:
            task.exception()
        except BaseException:
            pass

    def register_lifecycle_task(self, task: asyncio.Task[Any]) -> None:
        """Track asynchronous cleanup that must finish before queue handoff."""
        with self._claim_handoff_lock:
            self._lifecycle_tasks.add(task)
        task.add_done_callback(self._discard_lifecycle_task)

    async def _wait_for_lifecycle_tasks(self) -> None:
        """Wait for registered cleanup before publishing a new owner."""
        while True:
            with self._claim_handoff_lock:
                tasks = tuple(self._lifecycle_tasks)
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    def attach_handoff_receiver(self, receiver: Any) -> None:
        """Attach the adapter that owns this queue's local projections."""
        with self._claim_handoff_lock:
            self._handoff_receiver_ref = weakref.ref(receiver)

    def _handoff_receiver(self) -> Any:
        with self._claim_handoff_lock:
            reference = self._handoff_receiver_ref
        return reference() if reference is not None else None

    def _notify_handoff(self, replacement: "DurableTelegramUpdateQueue") -> None:
        """Retire or transfer predecessor-only adapter projections."""
        receiver = self._handoff_receiver()
        callback = getattr(receiver, "_on_inbound_queue_handoff", None)
        if callable(callback):
            callback(replacement)

    async def _wait_for_claim_event(self, event: asyncio.Event) -> None:
        """Wait in the event loop without consuming a default-executor worker."""
        await event.wait()

    async def _acquire_claim_operation(
        self, update_id: Optional[int]
    ) -> tuple[Optional[InboundEvent], Optional["DurableTelegramUpdateQueue"]]:
        """Reserve a claim operation, waiting for handoff or another operation."""
        if update_id is None:
            return None, None
        normalized_update_id = int(update_id)
        while True:
            wait_event = None
            with self._claim_handoff_lock:
                if self._handoff_in_progress:
                    wait_event = self._handoff_complete
                    claim = None
                    target = None
                else:
                    claim = self._claim_by_update_id.get(normalized_update_id)
                    if claim is None:
                        claim = self._retryable_claim_by_update_id.pop(
                            normalized_update_id, None
                        )
                        if claim is not None:
                            self._claim_by_update_id[normalized_update_id] = claim
                    target = self._handoff_target if claim is None else None
                    if claim is not None:
                        wait_event = self._begin_claiming_locked(claim.event_id)
            if claim is not None and wait_event is not None:
                return claim, None
            if claim is None and target is not None:
                return None, target
            if claim is None and target is None and wait_event is None:
                return None, None
            if wait_event is None:
                # Another operation owns the claim. Its completion event is
                # published while holding the same lock, so it cannot vanish.
                with self._claim_handoff_lock:
                    claim = self._claim_by_update_id.get(normalized_update_id)
                    wait_event = (
                        self._claiming_done.get(claim.event_id)
                        if claim is not None
                        else self._handoff_complete
                    )
                if wait_event is None:
                    continue
            await self._wait_for_claim_event(wait_event)

    async def mark_control_started(self, update_id: Optional[int]) -> bool:
        claim, target = await self._acquire_claim_operation(update_id)
        if claim is None:
            if target is not None:
                return await target.mark_control_started(update_id)
            return False
        operation = asyncio.create_task(
            self._run_store(
                self.run_claim_operation,
                claim,
                lambda current: self.store.mark_control_started(
                    current.event_id,
                    owner=current.lease_owner or self.lease_owner,
                    lease_epoch=current.lease_epoch,
                ),
            )
        )
        try:
            return bool(await asyncio.shield(operation))
        except asyncio.CancelledError:
            await asyncio.shield(operation)
            raise
        finally:
            self._finish_claiming(claim.event_id)

    async def complete_update(
        self,
        update_id: Optional[int],
        *,
        success: bool,
        delay: float = 0.0,
        defer: bool = False,
        _retrying: bool = False,
    ) -> bool:
        """Fence a completion transition until SQLite confirms its outcome."""
        claim, target = await self._acquire_claim_operation(update_id)
        if claim is None:
            if target is not None:
                return await target.complete_update(
                    update_id,
                    success=success,
                    delay=delay,
                    defer=defer,
                    _retrying=_retrying,
                )
            return False

        def transition(current: InboundEvent) -> bool:
            owner = current.lease_owner or self.lease_owner
            if success:
                return self.store.mark_consumed(
                    current.event_id,
                    owner=owner,
                    lease_epoch=current.lease_epoch,
                )
            if defer:
                return self.store.requeue(
                    current.event_id,
                    owner=owner,
                    lease_epoch=current.lease_epoch,
                    error_class="busy_cap",
                    delay=delay,
                    preserve_retry_budget=True,
                )
            return self.store.requeue(
                current.event_id,
                owner=owner,
                lease_epoch=current.lease_epoch,
                error_class="handler_failed",
                delay=delay,
            )

        operation = asyncio.create_task(
            self._run_store(self.run_claim_operation, claim, transition)
        )
        changed = False
        transition_error: Optional[Exception] = None
        cancelled = False
        try:
            changed = bool(await asyncio.shield(operation))
        except asyncio.CancelledError:
            cancelled = True
            try:
                changed = bool(await asyncio.shield(operation))
            except Exception as exc:
                transition_error = exc
        except Exception as exc:
            transition_error = exc

        retained = False
        if not changed:
            retained = await self._claim_still_owned(claim)
            if transition_error is not None:
                logger.warning(
                    "Telegram durable completion retry: bot_account_id=%s "
                    "event_id=%s error=%s",
                    self.bot_account_id,
                    claim.event_id,
                    type(transition_error).__name__,
                )
        with self._claim_handoff_lock:
            normalized_update_id = int(claim.update_id)
            if changed or not retained:
                self._buffered_event_ids.discard(claim.event_id)
                if self._claim_by_update_id.get(normalized_update_id) is claim:
                    self._claim_by_update_id.pop(normalized_update_id, None)
                    self._handler_event_ids.discard(claim.event_id)
                self._retryable_claim_by_update_id.pop(normalized_update_id, None)
            else:
                # Do not release local capacity or handler ownership until an
                # idempotent terminal/requeue transition is confirmed.
                self._claim_by_update_id[normalized_update_id] = claim
        self._finish_claiming(claim.event_id)
        if changed:
            self._request_wakeup()
        elif retained and not _retrying:
            self._schedule_claim_transition_retry(
                int(claim.update_id), success=success, delay=delay, defer=defer
            )
        if cancelled:
            raise asyncio.CancelledError
        return changed

    async def _claim_still_owned(self, claim: InboundEvent) -> bool:
        """Return whether an unconfirmed transition still owns this exact lease."""
        try:
            current = await self._run_store(self.store.get, claim.event_id)
        except Exception:
            # A second SQLite fault is not proof that the handler's completion
            # lost ownership. Retain the capacity and retry conservatively.
            return True
        return bool(
            current is not None
            and current.work_state == "leased"
            and current.lease_owner == claim.lease_owner
            and current.lease_epoch == claim.lease_epoch
        )

    def _schedule_claim_transition_retry(
        self, update_id: int, *, success: bool, delay: float, defer: bool
    ) -> None:
        """Retry only the SQLite completion, never the already-run handler."""
        existing = self._claim_transition_tasks.get(update_id)
        if existing is not None and not existing.done():
            return

        async def retry() -> None:
            retry_delay = max(0.0, float(self._persist_retry_initial_seconds))
            while True:
                await asyncio.sleep(retry_delay)
                try:
                    changed = await self.complete_update(
                        update_id,
                        success=success,
                        delay=delay,
                        defer=defer,
                        _retrying=True,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Telegram durable completion retry failed: "
                        "bot_account_id=%s update_id=%s error=%s",
                        self.bot_account_id,
                        update_id,
                        type(exc).__name__,
                    )
                    changed = False
                with self._claim_handoff_lock:
                    still_owned = update_id in self._claim_by_update_id
                if changed or not still_owned:
                    return
                retry_delay = min(
                    max(retry_delay * 2.0, self._persist_retry_initial_seconds),
                    self._persist_retry_max_seconds,
                )

        task = asyncio.create_task(retry())
        self._claim_transition_tasks[update_id] = task

        def discard(completed: asyncio.Task[Any]) -> None:
            if self._claim_transition_tasks.get(update_id) is completed:
                self._claim_transition_tasks.pop(update_id, None)
            try:
                completed.exception()
            except BaseException:
                pass

        task.add_done_callback(discard)

    def _cancel_due_wakeup(self) -> None:
        handle = self._due_wakeup_handle
        if handle is not None:
            handle.cancel()
        self._due_wakeup_handle = None
        self._due_wakeup_at = None
        task = self._due_wakeup_task
        if (
            task is not None
            and not task.done()
            and task is not asyncio.current_task()
        ):
            task.cancel()
            self._due_wakeup_task = None

    def _due_wakeup_fired(self) -> None:
        self._due_wakeup_handle = None
        self._due_wakeup_at = None
        if self._projection_suspended or self._lifecycle_retired:
            return
        loop = self._scheduler_loop
        if loop is None or loop.is_closed():
            return

        async def wake() -> None:
            try:
                await self.wake_scheduler()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Telegram durable due wake failed: bot_account_id=%s error=%s",
                    self.bot_account_id,
                    type(exc).__name__,
                )
                self._schedule_projection_retry()

        task = loop.create_task(wake())
        self._due_wakeup_task = task
        self._track_ingress_task(task)

        def clear_due_wakeup(completed: asyncio.Task[Any]) -> None:
            if self._due_wakeup_task is completed:
                self._due_wakeup_task = None

        task.add_done_callback(clear_due_wakeup)

    def _arm_due_wakeup(self, due_at: Optional[float]) -> None:
        loop = self._scheduler_loop
        if (
            self._projection_suspended
            or self._lifecycle_retired
            or loop is None
            or loop.is_closed()
            or due_at is None
        ):
            self._cancel_due_wakeup()
            return
        due = float(due_at)
        current_handle = self._due_wakeup_handle
        current_due = self._due_wakeup_at
        if (
            current_handle is not None
            and not current_handle.cancelled()
            and current_due is not None
            and current_due <= due
        ):
            return
        self._cancel_due_wakeup()
        self._due_wakeup_at = due
        self._due_wakeup_handle = loop.call_later(
            max(0.0, due - time.time()), self._due_wakeup_fired
        )

    async def _refresh_due_wakeup(self, *, capacity_available: bool) -> None:
        with self._claim_handoff_lock:
            protected_event_ids = self._handler_event_ids | self._buffered_event_ids
            lease_deadlines = tuple(
                claim.lease_expires_at
                for claim in self._claim_by_update_id.values()
                if (
                    claim.event_id not in protected_event_ids
                    and claim.lease_expires_at is not None
                )
            )
        due_at = min(lease_deadlines) if lease_deadlines else None
        if capacity_available:
            try:
                pending_due_at = await self._run_store(
                    self.store.next_pending_dispatch_at,
                    bot_account_id=self.bot_account_id,
                )
            except Exception:
                pending_due_at = None
            if pending_due_at is not None:
                due_at = (
                    pending_due_at
                    if due_at is None
                    else min(due_at, pending_due_at)
                )
        self._arm_due_wakeup(due_at)

    async def wake_scheduler(self) -> int:
        """Project a bounded, account-scoped durable backlog into PTB."""
        self._scheduler_loop = asyncio.get_running_loop()
        with self._claim_handoff_lock:
            target = self._handoff_target
        if target is not None:
            self._cancel_due_wakeup()
            return await target.wake_scheduler()
        if self._projection_suspended:
            self._cancel_due_wakeup()
            return 0

        projected = 0
        capacity_available = False
        try:
            async with self._admission_lock:
                # A handler may never be reached (e.g. an invalid projection
                # or a PTB group that does not match it). Reclaim this owner's
                # expired finite pre-handler leases before calculating capacity,
                # but never expire a handler or deliberate batch that still has
                # an in-memory completion driver.
                with self._claim_handoff_lock:
                    protected_event_ids = tuple(
                        self._handler_event_ids | self._buffered_event_ids
                    )
                await self._run_store(
                    self.store.reclaim_process_leases,
                    bot_account_id=self.bot_account_id,
                    current_owner=self.lease_owner,
                    exclude_event_ids=protected_event_ids,
                )
                now = time.time()
                with self._claim_handoff_lock:
                    for update_id, claim in tuple(self._claim_by_update_id.items()):
                        if (
                            claim.event_id not in protected_event_ids
                            and claim.lease_expires_at is not None
                            and claim.lease_expires_at <= now
                        ):
                            self._claim_by_update_id.pop(update_id, None)
                            self._retryable_claim_by_update_id.pop(update_id, None)
                            self._handler_event_ids.discard(claim.event_id)
                            self._buffered_event_ids.discard(claim.event_id)
                pending_resets = tuple(self._pending_dispatch_resets)
                for event_id in pending_resets:
                    try:
                        await self._run_store(
                            self.store.reset_dispatch_pending, event_id
                        )
                    except Exception:
                        self._schedule_projection_retry()
                        return 0
                    self._pending_dispatch_resets.discard(event_id)
                with self._claim_handoff_lock:
                    if self._handoff_in_progress:
                        capacity = 0
                    else:
                        active = (
                            len(self._queued_event_ids)
                            + len(self._claiming_event_ids)
                            + len(self._claim_by_update_id)
                            + len(self._retryable_claim_by_update_id)
                        )
                        capacity = self.active_limit - active
                capacity_available = capacity > 0
                if capacity <= 0:
                    return 0
                rows = await self._run_store(
                    self.store.pending_dispatch,
                    bot_account_id=self.bot_account_id,
                    limit=capacity,
                )
                for row in rows:
                    with self._claim_handoff_lock:
                        if self._handoff_in_progress:
                            break
                        already_queued = row.event_id in self._queued_event_ids
                    if row.payload is None or already_queued:
                        continue
                    admitted = await self._run_store(
                        self.store.mark_dispatch_admitted, row.event_id
                    )
                    if not admitted:
                        continue
                    update_id = self._update_id(row.payload, row.payload)
                    try:
                        if update_id is None:
                            raise ValueError("durable payload has no integer update_id")
                        item = self.item_factory(row.payload)
                        if not self._is_update(item):
                            raise ValueError(
                                "durable item factory did not return a Telegram update"
                            )
                        with self._claim_handoff_lock:
                            if self._handoff_in_progress:
                                raise RuntimeError(
                                    "Telegram queue handoff started during admission"
                                )
                            self._queued_event_ids.add(row.event_id)
                            self._event_id_by_update_id[update_id] = row.event_id
                            self.put_nowait(item)
                    except Exception as exc:
                        logger.warning(
                            "Telegram durable projection rejected: bot_account_id=%s "
                            "event_id=%s error=%s",
                            self.bot_account_id,
                            row.event_id,
                            type(exc).__name__,
                        )
                        with self._claim_handoff_lock:
                            self._queued_event_ids.discard(row.event_id)
                            if update_id is not None:
                                self._event_id_by_update_id.pop(update_id, None)
                        try:
                            await self._run_store(
                                self.store.reset_dispatch_pending, row.event_id
                            )
                        except Exception:
                            self._pending_dispatch_resets.add(row.event_id)
                            self._schedule_projection_retry()
                        # A factory/configuration fault cannot be made healthy
                        # by immediately reprojecting the same row in a timer
                        # loop. Keep it replayable for explicit recovery.
                        capacity_available = False
                        break
                    projected += 1
        finally:
            await self._refresh_due_wakeup(
                capacity_available=capacity_available
            )
        return projected

    async def recover(self, *, now: Optional[float] = None) -> int:
        """Recover this queue's account and project the replayable backlog."""
        self._projection_suspended = False
        await self._run_store(
            self.store.reclaim_process_leases,
            bot_account_id=self.bot_account_id,
            current_owner=self.lease_owner,
            now=now,
        )
        await self._run_store(
            self.store.recover_admitted_dispatches,
            bot_account_id=self.bot_account_id,
            now=now,
        )
        return await self.wake_scheduler()

    def _request_wakeup(self) -> None:
        if self._projection_suspended or self._lifecycle_retired:
            return
        loop = self._scheduler_loop
        if loop is None or loop.is_closed():
            return

        loop.call_soon_threadsafe(self._schedule_projection_retry)

    async def suspend_projection(self) -> None:
        """Stop queue-owned projection tasks until lifecycle recovery resumes."""
        self._projection_suspended = True
        due_wakeup_task = self._due_wakeup_task
        self._cancel_due_wakeup()
        if (
            due_wakeup_task is not None
            and not due_wakeup_task.done()
            and due_wakeup_task is not asyncio.current_task()
        ):
            await asyncio.gather(due_wakeup_task, return_exceptions=True)
        # Flush any call_soon_threadsafe callback queued by complete_update().
        # It observes the suspension flag and exits without creating a task.
        await asyncio.sleep(0)
        task = self._projection_retry_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if task is None or task.done():
            self._projection_retry_task = None

    def _shutdown_store_executor(self) -> None:
        """Retire this queue's worker without abandoning submitted writes."""
        if self._store_executor_shutdown:
            return
        self._store_executor_shutdown = True
        _release_store_executor(self._store_executor_key, self._store_executor)

    async def _retire_after_handoff(self) -> None:
        """Release queue-local workers after durable ownership transfers."""
        await self.suspend_projection()
        self._shutdown_store_executor()

    async def close(self) -> None:
        """Release queue-owned timers, retries, and its shared SQLite reference."""
        async with self._ingress_handoff_lock:
            self._ingress_closed = True
            self._lifecycle_retired = True
            self._handoff_complete.set()
        try:
            await self.suspend_projection()
            await self._wait_for_ingress_tasks()
            for task in tuple(self._claim_transition_tasks.values()):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()
            if self._claim_transition_tasks:
                await asyncio.gather(
                    *tuple(self._claim_transition_tasks.values()), return_exceptions=True
                )
            self._claim_transition_tasks.clear()
        finally:
            self._shutdown_store_executor()

    def forget_claims(self, event_ids: set[str]) -> None:
        if not event_ids:
            return
        released = False
        with self._claim_handoff_lock:
            for update_id, claim in tuple(self._claim_by_update_id.items()):
                if claim.event_id in event_ids:
                    self._claim_by_update_id.pop(update_id, None)
                    self._handler_event_ids.discard(claim.event_id)
                    self._buffered_event_ids.discard(claim.event_id)
                    released = True
            for update_id, claim in tuple(self._retryable_claim_by_update_id.items()):
                if claim.event_id in event_ids:
                    self._retryable_claim_by_update_id.pop(update_id, None)
                    self._buffered_event_ids.discard(claim.event_id)
                    released = True
        if released:
            self._request_wakeup()

    async def put(self, item: Any) -> None:
        target = await self._enter_ingress()
        if target is not None:
            await target.put(item)
            return
        try:
            raw = self._raw_payload(item)
            if not self._is_update(item, raw):
                # PTB uses a private object sentinel during shutdown. It is not
                # user work and must retain ordinary asyncio.Queue semantics.
                await super().put(item)
                return
            decision = await self._classify_with_retry(item, raw)
            if not decision.actionable:
                await super().put(item)
                return
            update_id = self._update_id(item, raw)
            assert update_id is not None
            persist_task = asyncio.create_task(
                self._persist_with_retry(update_id=update_id, decision=decision)
            )
            try:
                result = await asyncio.shield(persist_task)
            except asyncio.CancelledError:
                # A reconnect must not wait forever for a worker thread that cannot
                # be cancelled. PTB has not advanced its offset, while a late commit
                # remains idempotent and is projected by this background recovery.
                recovery = asyncio.create_task(
                    self._recover_committed_put(item, persist_task)
                )
                self._track_ingress_task(recovery)
                raise
            await self._admit_result(item, result)
        finally:
            self._leave_ingress()

    async def _recover_committed_put(
        self, item: Any, task: asyncio.Task[PersistResult]
    ) -> None:
        try:
            result = await task
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                "Telegram durable ingress background recovery failed: "
                "bot_account_id=%s error=%s",
                self.bot_account_id,
                type(exc).__name__,
            )
            return
        await self._admit_result(item, result)

    async def _admit_result(self, item: Any, result: PersistResult) -> None:
        with self._claim_handoff_lock:
            target = self._handoff_target
        if target is not None:
            # This put began in the predecessor epoch but committed after the
            # replacement published. The row is already durable; projection is
            # now solely the replacement's responsibility.
            try:
                await target.wake_scheduler()
            except asyncio.CancelledError:
                # The committed row still needs a projection attempt after PTB
                # stops awaiting this ingress put.
                target._schedule_projection_retry()
                raise
            return
        try:
            await self.wake_scheduler()
        except asyncio.CancelledError:
            # Persist succeeded before this cancellation. Retain an owned
            # projection retry so that a canceled PTB put cannot strand the
            # durable row until an unrelated reconnect happens.
            self._schedule_projection_retry()
            raise
        except Exception as exc:
            # The Bot API update is already durable. Do not let projection or
            # callback failures kill PTB's polling task; the durable row remains
            # replayable and the background scheduler retries projection.
            logger.warning(
                "Telegram durable ingress committed but projection was deferred: "
                "bot_account_id=%s event_id=%s error=%s",
                self.bot_account_id,
                result.event_id,
                type(exc).__name__,
            )
            self._schedule_projection_retry()
            return
        if result.duplicate or self.after_commit is None:
            return
        try:
            callback_result = self.after_commit(item, result)
            if callback_result is not None:
                await callback_result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Telegram durable ingress post-commit callback failed: "
                "bot_account_id=%s event_id=%s error=%s",
                self.bot_account_id,
                result.event_id,
                type(exc).__name__,
            )

    async def put_replay(self, item: Any, event_id: str) -> None:
        del item, event_id
        await self.wake_scheduler()

    async def get(self) -> Any:
        while True:
            item = await super().get()
            raw = self._raw_payload(item)
            update_id = self._update_id(item, raw)
            if update_id is None:
                return item

            event_id = None
            wait_event = None
            handoff_target = None
            with self._claim_handoff_lock:
                if self._handoff_in_progress:
                    # A durable projection keeps its mapping until the
                    # handoff publishes the replacement claim. Ordinary
                    # updates must retain their normal queue semantics.
                    event_id = self._event_id_by_update_id.get(update_id)
                    if event_id is not None:
                        wait_event = self._handoff_complete
                elif update_id in self._handed_off_update_ids:
                    self._handed_off_update_ids.discard(update_id)
                    handoff_target = self._handoff_target
                else:
                    event_id = self._event_id_by_update_id.pop(update_id, None)
                    if event_id is not None:
                        self._queued_event_ids.discard(event_id)

            if wait_event is not None:
                try:
                    await self._wait_for_claim_event(wait_event)
                except asyncio.CancelledError:
                    super().task_done()
                    raise
                # The predecessor's physical projection belongs to the retired
                # consumer.  Discard it rather than returning replacement work
                # through an old adapter whose wrapper no longer owns the claim.
                super().task_done()
                continue

            if handoff_target is not None:
                super().task_done()
                continue

            if event_id is None:
                return item

            with self._claim_handoff_lock:
                claim_done = self._begin_claiming_locked(event_id)
            if claim_done is None:
                with self._claim_handoff_lock:
                    claim_done = self._claiming_done.get(event_id)
                if claim_done is not None:
                    await self._wait_for_claim_event(claim_done)
                super().task_done()
                continue

            lease_task = asyncio.create_task(
                self._run_store(
                    self.store.lease_event,
                    event_id,
                    owner=self.lease_owner,
                    lease_seconds=self._prehandler_lease_seconds,
                )
            )
            leased = None
            try:
                leased = await asyncio.shield(lease_task)
            except asyncio.CancelledError:
                # Shield prevents cancellation from killing the worker thread,
                # but the SQLite lease may already have committed. Wait for
                # its result, then explicitly undo that lease before exposing
                # the cancellation to the caller.
                try:
                    try:
                        leased = await asyncio.shield(lease_task)
                    except BaseException:
                        leased = None
                    if leased is not None:
                        reverted = await self._run_store(
                            self.store.requeue,
                            event_id,
                            owner=self.lease_owner,
                            lease_epoch=leased.lease_epoch,
                            error_class="claim_cancelled",
                        )
                        if not reverted:
                            await self._run_store(
                                self.store.reset_dispatch_pending, event_id
                            )
                    else:
                        await self._run_store(
                            self.store.reset_dispatch_pending, event_id
                        )
                except BaseException:
                    # Preserve cancellation even when SQLite remains busy. A
                    # later lifecycle recovery owns authoritative cleanup.
                    pass
                finally:
                    self._discard_claiming_projection(event_id)
                try:
                    await self.wake_scheduler()
                except BaseException:
                    pass
                raise
            except Exception as exc:
                # PTB's update fetcher awaits queue.get() outside its recovery
                # loop. A transient SQLite fault must therefore release this
                # projection and retry locally rather than terminate fetching.
                logger.warning(
                    "Telegram durable lease retry: bot_account_id=%s event_id=%s error=%s",
                    self.bot_account_id,
                    event_id,
                    type(exc).__name__,
                )
                self._pending_dispatch_resets.add(event_id)
                try:
                    current = await self._run_store(self.store.get, event_id)
                    if (
                        current is not None
                        and current.work_state == "leased"
                        and current.lease_owner == self.lease_owner
                    ):
                        reverted = await self._run_store(
                            self.store.requeue,
                            event_id,
                            owner=self.lease_owner,
                            lease_epoch=current.lease_epoch,
                            error_class="claim_failed",
                        )
                        if reverted:
                            self._pending_dispatch_resets.discard(event_id)
                    else:
                        await self._run_store(
                            self.store.reset_dispatch_pending, event_id
                        )
                        self._pending_dispatch_resets.discard(event_id)
                except Exception:
                    # The next bounded projection wake/recovery remains the
                    # authority if the cleanup read is also transiently busy.
                    logger.warning(
                        "Telegram durable lease cleanup deferred: bot_account_id=%s event_id=%s",
                        self.bot_account_id,
                        event_id,
                    )
                finally:
                    self._discard_claiming_projection(event_id)
                try:
                    await self.wake_scheduler()
                except Exception:
                    self._schedule_projection_retry()
                await asyncio.sleep(
                    max(0.0, float(self._persist_retry_initial_seconds))
                )
                continue

            if leased is None:
                try:
                    await self._run_store(
                        self.store.reset_dispatch_pending, event_id
                    )
                finally:
                    self._discard_claiming_projection(event_id)
                await self.wake_scheduler()
                continue

            # Publish the claim while the operation fence is still held. A
            # handoff that started concurrently therefore observes either this
            # complete claim or the cleanup path above, never an invisible
            # SQLite lease.
            with self._claim_handoff_lock:
                self._claim_by_update_id[update_id] = leased
            self._finish_claiming(event_id)
            # No handler has claimed this row yet. Arm a finite recovery wake so
            # an unmatched PTB update cannot permanently occupy active_limit.
            await self._refresh_due_wakeup(capacity_available=False)
            return item

    async def handoff_from(self, previous: "DurableTelegramUpdateQueue") -> dict[str, int]:
        """Transfer live claims and make abandoned projections replayable."""
        if previous.store.lifecycle_path != self.store.lifecycle_path:
            raise ValueError("Telegram queue handoff requires the same store")
        if previous.bot_account_id != self.bot_account_id:
            raise ValueError("Telegram queue handoff requires the same bot account")
        self._claim_handoff_lock = previous._claim_handoff_lock

        # Close admission before claiming predecessor state. New old-queue puts
        # wait for publication and then forward to this queue; already-admitted
        # puts keep their shared executor and redirect their post-commit wake
        # below. Handoff therefore remains bounded during a transient outage.
        async with previous._ingress_handoff_lock:
            with previous._claim_handoff_lock:
                if previous._handoff_in_progress:
                    raise RuntimeError("Telegram queue handoff already in progress")
                previous._ingress_closed = True
                previous._handoff_complete.clear()

        async with previous._admission_lock:
            # Held-event overflow cleanup is scheduled from a synchronous
            # adapter path. Finish it before classifying claims as live, or the
            # replacement could transfer a lease that the old adapter is about
            # to release.
            await previous._wait_for_lifecycle_tasks()
            with previous._claim_handoff_lock:
                previous._handoff_in_progress = True
                # Claims already inside get() when handoff began are live even
                # if the handler has not received the returned item yet.
                # Previously completed, unhandled claims remain replayable.
                handoff_live_event_ids = set(previous._handler_event_ids)
                handoff_live_event_ids.update(previous._claiming_event_ids)
                handoff_live_event_ids.difference_update(
                    previous._buffered_event_ids
                )
                claim_wait_events = tuple(previous._claiming_done.values())

            try:
                for claim_wait_event in claim_wait_events:
                    await previous._wait_for_claim_event(claim_wait_event)
                while True:
                    with previous._claim_handoff_lock:
                        remaining = tuple(previous._claiming_done.values())
                    if not remaining:
                        break
                    for claim_wait_event in remaining:
                        await previous._wait_for_claim_event(claim_wait_event)

                def handoff_and_publish() -> dict[str, int]:
                    with previous._claim_handoff_lock:
                        handoff_live_event_ids.update(previous._handler_event_ids)
                        handoff_live_event_ids.difference_update(
                            previous._buffered_event_ids
                        )
                        claims = tuple(previous._claim_by_update_id.items())
                        counts = self.store.handoff_owner_leases(
                            bot_account_id=self.bot_account_id,
                            old_owner=previous.lease_owner,
                            new_owner=self.lease_owner,
                            live_event_ids=handoff_live_event_ids,
                        )
                        previous._fenced_handler_update_ids.update(
                            update_id for update_id, _claim in claims
                        )
                        for event_id in tuple(previous._queued_event_ids):
                            self.store.reset_dispatch_pending(event_id)
                        for update_id, claim in claims:
                            if claim.event_id not in handoff_live_event_ids:
                                continue
                            current = self.store.get(claim.event_id)
                            if (
                                current is not None
                                and current.work_state == "leased"
                                and current.lease_owner == self.lease_owner
                                and current.lease_epoch == claim.lease_epoch
                            ):
                                claim.lease_owner = current.lease_owner
                                self._claim_by_update_id[update_id] = claim
                                self._handler_event_ids.add(claim.event_id)
                        previous._claim_by_update_id.clear()
                        previous._retryable_claim_by_update_id.clear()
                        previous._handler_event_ids.clear()
                        previous._buffered_event_ids.clear()
                        previous._handed_off_update_ids.update(
                            previous._event_id_by_update_id
                        )
                        previous._event_id_by_update_id.clear()
                        previous._queued_event_ids.clear()
                        previous._handoff_target = self
                        previous._lifecycle_retired = True
                        previous._handoff_in_progress = False
                        previous._handoff_complete.set()
                        return counts

                counts = await self._run_store(handoff_and_publish)
            finally:
                with previous._claim_handoff_lock:
                    if previous._handoff_in_progress:
                        previous._handoff_in_progress = False
                        previous._ingress_closed = False
                        previous._handoff_complete.set()

        # A hold can be created while the threaded handoff is running. Such a
        # cleanup redirects to this queue after publication; wait for it before
        # the replacement's final projection wake.
        await previous._wait_for_lifecycle_tasks()
        await previous._retire_after_handoff()
        previous._notify_handoff(self)
        await self.wake_scheduler()
        return counts


    async def _requeue_unclaimed_projection_ids(
        self, previous: "DurableTelegramUpdateQueue"
    ) -> int:
        del previous
        return 0


class TelegramQueueLifecycleRegistry:
    """Serialize durable queue recovery for each store and bot account."""

    _registry_lock = threading.RLock()
    _active: dict[tuple[str, str], weakref.ReferenceType] = {}
    _transition_locks: dict[tuple[str, str], threading.Lock] = {}

    @classmethod
    def key_for(
        cls, store_path: os.PathLike[str] | str, bot_account_id: Any
    ) -> tuple[str, str]:
        """Return one process-local key for each physical SQLite path."""
        return (
            _canonical_store_path(store_path),
            canonical_bot_account_id(bot_account_id),
        )

    @classmethod
    def key_for_queue(
        cls, queue: DurableTelegramUpdateQueue
    ) -> tuple[str, str]:
        return cls.key_for(queue.store.lifecycle_path, queue.bot_account_id)

    @classmethod
    def _active_queue_locked(
        cls, key: tuple[str, str]
    ) -> Optional[DurableTelegramUpdateQueue]:
        reference = cls._active.get(key)
        if reference is None:
            return None
        queue = reference()
        if queue is None:
            cls._active.pop(key, None)
        return queue

    @classmethod
    def observe(cls, queue: DurableTelegramUpdateQueue) -> None:
        """Register a queue without replacing an already active queue."""
        key = cls.key_for_queue(queue)
        with cls._registry_lock:
            if cls._active_queue_locked(key) is None:
                cls._active[key] = weakref.ref(queue)

    @classmethod
    def _transition_lock_for(cls, key: tuple[str, str]) -> threading.Lock:
        with cls._registry_lock:
            return cls._transition_locks.setdefault(key, threading.Lock())

    @staticmethod
    async def _acquire_transition_lock(lock: threading.Lock) -> None:
        """Acquire without parking an executor thread that cancellation leaks."""
        while not lock.acquire(blocking=False):
            await asyncio.sleep(0.001)

    @classmethod
    async def recover(
        cls, queue: DurableTelegramUpdateQueue, *, now: Optional[float] = None
    ) -> int:
        """Handoff a replaced queue before recovering the replacement."""
        key = cls.key_for_queue(queue)
        transition_lock = cls._transition_lock_for(key)
        await cls._acquire_transition_lock(transition_lock)
        try:
            with cls._registry_lock:
                active = cls._active_queue_locked(key)
                if getattr(queue, "_lifecycle_retired", False):
                    return 0
                if active is None:
                    cls._active[key] = weakref.ref(queue)
                    active = queue

            if active is not queue:
                handoff_already_complete = (
                    getattr(active, "_lifecycle_retired", False)
                    and getattr(active, "_handoff_target", None) is queue
                )
                if not handoff_already_complete:
                    await queue.handoff_from(active)
                with cls._registry_lock:
                    cls._active[key] = weakref.ref(queue)

            return await queue.recover(now=now)
        finally:
            transition_lock.release()
