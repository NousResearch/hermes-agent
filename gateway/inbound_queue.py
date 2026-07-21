"""Durable, fair inbound-turn storage for the messaging Gateway.

The Gateway's session database is intentionally *not* used here.  Inbound
turns are the recovery source of truth when the main transcript database is
busy or the process is killed before turn-start persistence completes, so
sharing that database would couple the recovery path to the failure it is
supposed to survive.

This module owns storage only.  Dispatch, authorization, and recovery policy
live in :mod:`gateway.run`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_FILENAME = "gateway-inbox.db"
_SCHEMA_VERSION = 1
_DEFAULT_BUSY_TIMEOUT_MS = 5_000
_DEFAULT_MAX_PENDING_PER_SESSION = 64
_DEFAULT_MAX_PENDING_TOTAL = 8_192
_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_TERMINAL_RETENTION_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_MAX_ROWS = 50_000
_DEFAULT_PRUNE_INTERVAL_SECONDS = 60.0
_MAX_PAYLOAD_BYTES = 1_000_000

_PENDING_STATES = ("queued", "claimed", "resume_ready")
_FINAL_STATES = ("completed", "cancelled", "dead_letter")
_ALL_STATES = _PENDING_STATES + _FINAL_STATES
_ORIGINS = frozenset({"direct", "busy", "explicit", "startup"})

EXPLICIT_QUEUE_METADATA_KEY = "_hermes_explicit_queue"
INBOX_METADATA_KEY = "_hermes_gateway_inbox"

# Only routing/context values consumed after a queued turn is restored belong
# here.  Arbitrary adapter objects and trust-bearing private state must never
# be serialized into the recovery database.
_EVENT_METADATA_KEYS = frozenset({
    EXPLICIT_QUEUE_METADATA_KEY,
    "channel_id",
    "context_channel_id",
    "direct_messages_topic_id",
    "display_phone_number",
    "gateway_session_id",
    "message_thread_id",
    "non_conversational",
    "notify",
    "placeholder_text",
    "publish_topic",
    "reply_to_message_id",
    "root_id",
    "slack_team_id",
    "team",
    "team_id",
    "telegram_direct_messages_topic_id",
    "telegram_dm_topic_created_for_send",
    "telegram_dm_topic_reply_fallback",
    "telegram_reply_to_message_id",
    "thread_id",
    "thread_name",
    "thread_ts",
    "user_id",
    "whatsapp_native",
    "whatsapp_native_type",
})

_EXPLICIT_QUEUE_METADATA_KEYS = frozenset({
    "id",
    "owner_user_id",
    "created_at",
    "origin",
})

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_WRITER_LOCKS: dict[str, threading.RLock] = {}


class InboxSerializationError(ValueError):
    """A MessageEvent cannot be represented by the durable wire format."""


class InboxPayloadError(ValueError):
    """A stored payload is malformed or from an unsupported schema version."""


@dataclass(frozen=True)
class InboxRow:
    """Decoded durable queue row."""

    id: int
    queue_id: str
    dedupe_key: str
    session_key: str
    session_id: Optional[str]
    profile: Optional[str]
    platform: str
    trigger_identity: str
    payload: Mapping[str, Any]
    origin: str
    state: str
    priority: int
    attempts: int
    owner_pid: Optional[int]
    owner_started_at: Optional[int]
    claim_token: Optional[str]
    resume_only: bool
    created_at: float
    updated_at: float
    claimed_at: Optional[float]
    completed_at: Optional[float]
    not_before: float
    last_error: Optional[str]

    def to_event(self):
        """Rebuild a safe MessageEvent with an ephemeral inbox claim marker."""
        event = deserialize_message_event(self.payload)
        metadata = dict(getattr(event, "metadata", None) or {})
        metadata[INBOX_METADATA_KEY] = {
            "queue_id": self.queue_id,
            "claim_token": self.claim_token,
            "trigger_identity": self.trigger_identity,
            "session_key": self.session_key,
            "resume_only": self.resume_only,
        }
        event.metadata = metadata
        return event


@dataclass(frozen=True)
class EnqueueResult:
    """Result of an idempotent enqueue attempt."""

    accepted: bool
    inserted: bool
    status: str
    row: Optional[InboxRow] = None


def _process_writer_lock(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_WRITER_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_WRITER_LOCKS[key] = lock
        return lock


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """Return whether a pid still identifies the process that claimed a row."""
    if pid is None:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_started_at = get_process_start_time(pid)
    except Exception:
        current_started_at = None
    if current_started_at is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    if started_at is None:
        return True
    try:
        return int(started_at) == int(current_started_at)
    except (TypeError, ValueError):
        return True


def _is_corruption_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "database disk image is malformed",
            "database corruption",
            "file is encrypted",
            "file is not a database",
            "malformed database schema",
            "not a database",
        )
    )


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Project a metadata value onto bounded JSON primitives."""
    if depth > 8:
        raise TypeError("metadata nesting exceeds durable inbox limit")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in value.items()
            if isinstance(key, str)
        }
    raise TypeError(f"unsupported metadata type: {type(value).__name__}")


def _filtered_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    result: dict[str, Any] = {}
    for key in _EVENT_METADATA_KEYS:
        if key not in metadata:
            continue
        try:
            value = metadata[key]
            if key == EXPLICIT_QUEUE_METADATA_KEY:
                if not isinstance(value, dict):
                    continue
                value = {
                    nested_key: value[nested_key]
                    for nested_key in _EXPLICIT_QUEUE_METADATA_KEYS
                    if nested_key in value
                }
            result[key] = _json_safe(value)
        except (TypeError, ValueError, OverflowError):
            logger.debug("Skipping non-serializable inbound metadata key %s", key)
    return result


def serialize_message_event(event: Any) -> dict[str, Any]:
    """Serialize MessageEvent and SessionSource through an explicit whitelist."""
    from gateway.platforms.base import MessageType

    source = getattr(event, "source", None)
    if source is None:
        raise InboxSerializationError("inbound event has no SessionSource")
    platform = getattr(source, "platform", None)
    platform_value = getattr(platform, "value", platform)
    if not isinstance(platform_value, str) or not platform_value:
        raise InboxSerializationError("inbound event has no valid platform")
    message_type = getattr(event, "message_type", MessageType.TEXT)
    message_type_value = getattr(message_type, "value", message_type)
    if not isinstance(message_type_value, str):
        raise InboxSerializationError("inbound event has no valid message type")
    timestamp = getattr(event, "timestamp", None)
    if not isinstance(timestamp, datetime):
        raise InboxSerializationError("inbound event timestamp must be datetime")

    auto_skill = getattr(event, "auto_skill", None)
    if isinstance(auto_skill, list):
        auto_skill = [str(item) for item in auto_skill]
    elif auto_skill is not None:
        auto_skill = str(auto_skill)

    payload = {
        "version": _SCHEMA_VERSION,
        "source": {
            "platform": platform_value,
            "chat_id": str(getattr(source, "chat_id", "")),
            "chat_name": getattr(source, "chat_name", None),
            "chat_type": str(getattr(source, "chat_type", "dm") or "dm"),
            "user_id": getattr(source, "user_id", None),
            "user_name": getattr(source, "user_name", None),
            "thread_id": getattr(source, "thread_id", None),
            "chat_topic": getattr(source, "chat_topic", None),
            "user_id_alt": getattr(source, "user_id_alt", None),
            "chat_id_alt": getattr(source, "chat_id_alt", None),
            "is_bot": bool(getattr(source, "is_bot", False)),
            "scope_id": getattr(source, "scope_id", None),
            "parent_chat_id": getattr(source, "parent_chat_id", None),
            "message_id": getattr(source, "message_id", None),
            "profile": getattr(source, "profile", None),
            "auto_thread_created": bool(getattr(source, "auto_thread_created", False)),
            "auto_thread_initial_name": getattr(
                source, "auto_thread_initial_name", None
            ),
        },
        "event": {
            "text": str(getattr(event, "text", "") or ""),
            "message_type": message_type_value,
            "message_id": (
                str(event.message_id)
                if getattr(event, "message_id", None) is not None
                else None
            ),
            "platform_update_id": getattr(event, "platform_update_id", None),
            "media_urls": [
                str(item) for item in (getattr(event, "media_urls", None) or [])
            ],
            "media_types": [
                str(item) for item in (getattr(event, "media_types", None) or [])
            ],
            "reply_to_message_id": getattr(event, "reply_to_message_id", None),
            "reply_to_text": getattr(event, "reply_to_text", None),
            "reply_to_author_id": getattr(event, "reply_to_author_id", None),
            "reply_to_author_name": getattr(event, "reply_to_author_name", None),
            "reply_to_is_own_message": bool(
                getattr(event, "reply_to_is_own_message", False)
            ),
            "auto_skill": auto_skill,
            "channel_prompt": getattr(event, "channel_prompt", None),
            "channel_context": getattr(event, "channel_context", None),
            "internal": bool(getattr(event, "internal", False)),
            "metadata": _filtered_metadata(getattr(event, "metadata", None)),
            "timestamp": timestamp.isoformat(),
        },
    }
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InboxSerializationError(str(exc)) from exc
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise InboxSerializationError(
            f"inbound event payload exceeds {_MAX_PAYLOAD_BYTES} bytes"
        )
    return payload


def deserialize_message_event(payload: Mapping[str, Any]):
    """Reconstruct a MessageEvent from the versioned whitelist payload."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    if not isinstance(payload, Mapping) or payload.get("version") != _SCHEMA_VERSION:
        raise InboxPayloadError("unsupported durable inbox payload version")
    source_data = payload.get("source")
    event_data = payload.get("event")
    if not isinstance(source_data, Mapping) or not isinstance(event_data, Mapping):
        raise InboxPayloadError("durable inbox payload is missing event/source")
    try:
        source = SessionSource(
            platform=Platform(str(source_data["platform"])),
            chat_id=str(source_data["chat_id"]),
            chat_name=source_data.get("chat_name"),
            chat_type=str(source_data.get("chat_type") or "dm"),
            user_id=source_data.get("user_id"),
            user_name=source_data.get("user_name"),
            thread_id=source_data.get("thread_id"),
            chat_topic=source_data.get("chat_topic"),
            user_id_alt=source_data.get("user_id_alt"),
            chat_id_alt=source_data.get("chat_id_alt"),
            is_bot=bool(source_data.get("is_bot", False)),
            scope_id=source_data.get("scope_id"),
            parent_chat_id=source_data.get("parent_chat_id"),
            message_id=source_data.get("message_id"),
            profile=source_data.get("profile"),
            auto_thread_created=bool(source_data.get("auto_thread_created", False)),
            auto_thread_initial_name=source_data.get("auto_thread_initial_name"),
        )
        timestamp = datetime.fromisoformat(str(event_data["timestamp"]))
        metadata = event_data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return MessageEvent(
            text=str(event_data.get("text") or ""),
            message_type=MessageType(str(event_data.get("message_type") or "text")),
            source=source,
            raw_message=None,
            message_id=event_data.get("message_id"),
            platform_update_id=event_data.get("platform_update_id"),
            media_urls=[str(item) for item in (event_data.get("media_urls") or [])],
            media_types=[str(item) for item in (event_data.get("media_types") or [])],
            reply_to_message_id=event_data.get("reply_to_message_id"),
            reply_to_text=event_data.get("reply_to_text"),
            reply_to_author_id=event_data.get("reply_to_author_id"),
            reply_to_author_name=event_data.get("reply_to_author_name"),
            reply_to_is_own_message=bool(
                event_data.get("reply_to_is_own_message", False)
            ),
            auto_skill=event_data.get("auto_skill"),
            channel_prompt=event_data.get("channel_prompt"),
            channel_context=event_data.get("channel_context"),
            internal=bool(event_data.get("internal", False)),
            metadata=metadata,
            timestamp=timestamp,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InboxPayloadError(f"invalid durable inbox payload: {exc}") from exc


def _extract_explicit_queue_id(event: Any) -> Optional[str]:
    metadata = getattr(event, "metadata", None)
    marker = (
        metadata.get(EXPLICIT_QUEUE_METADATA_KEY)
        if isinstance(metadata, dict)
        else None
    )
    queue_id = marker.get("id") if isinstance(marker, dict) else None
    if not isinstance(queue_id, str):
        return None
    queue_id = queue_id.strip()
    return queue_id[:128] if queue_id else None


def _new_queue_id() -> str:
    return f"q-{secrets.token_hex(12)}"


def _new_claim_token() -> str:
    return secrets.token_urlsafe(24)


def _dedupe_key(event: Any, session_key: str, queue_id: str) -> str:
    message_id = getattr(event, "message_id", None)
    if message_id is None or not str(message_id).strip():
        return f"arrival:{queue_id}"
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", "")
    profile = getattr(source, "profile", None) or "default"
    material = "\x1f".join((
        str(platform),
        str(profile),
        str(session_key),
        str(message_id),
    ))
    digest = hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()
    return f"message:{digest}"


_SELECT_COLUMNS = """
    id, queue_id, dedupe_key, session_key, session_id, profile, platform,
    trigger_identity, payload_json, origin, state, priority, attempts,
    owner_pid, owner_started_at, claim_token, resume_only, created_at,
    updated_at, claimed_at, completed_at, not_before, last_error
"""


class GatewayInboxStore:
    """Small standalone SQLite queue with crash-safe ownership claims."""

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        *,
        max_pending_per_session: int = _DEFAULT_MAX_PENDING_PER_SESSION,
        max_pending_total: int = _DEFAULT_MAX_PENDING_TOTAL,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        terminal_retention_seconds: float = _DEFAULT_TERMINAL_RETENTION_SECONDS,
        max_rows: int = _DEFAULT_MAX_ROWS,
        prune_interval_seconds: float = _DEFAULT_PRUNE_INTERVAL_SECONDS,
    ) -> None:
        self._hermes_home = Path(hermes_home or get_hermes_home())
        self.max_pending_per_session = max(0, int(max_pending_per_session))
        self.max_pending_total = max(0, int(max_pending_total))
        self.max_attempts = max(1, int(max_attempts))
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.terminal_retention_seconds = max(0.0, float(terminal_retention_seconds))
        self.max_rows = max(1, int(max_rows))
        self.prune_interval_seconds = max(0.0, float(prune_interval_seconds))
        self._last_prune_monotonic = 0.0
        self._initialized = False
        self._writer_lock = _process_writer_lock(self.path())
        self._ensure_initialized()

    def path(self) -> Path:
        return self._hermes_home / "gateway" / _DB_FILENAME

    def close(self) -> None:
        """Compatibility no-op: operations intentionally use short connections."""

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path(),
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._writer_lock:
            if self._initialized:
                return
            path = self.path()
            path.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(2):
                conn: Optional[sqlite3.Connection] = None
                try:
                    conn = self._open()
                    journal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                    if str(journal_mode).lower() != "wal":
                        raise sqlite3.DatabaseError(
                            f"durable inbox refused WAL mode: {journal_mode}"
                        )
                    check = conn.execute("PRAGMA quick_check(1)").fetchone()
                    if check is not None and str(check[0]).lower() != "ok":
                        raise sqlite3.DatabaseError(
                            f"database disk image is malformed: {check[0]}"
                        )
                    conn.executescript(
                        f"""
                        CREATE TABLE IF NOT EXISTS inbound_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            queue_id TEXT NOT NULL UNIQUE,
                            dedupe_key TEXT NOT NULL UNIQUE,
                            session_key TEXT NOT NULL,
                            session_id TEXT,
                            profile TEXT,
                            platform TEXT NOT NULL,
                            trigger_identity TEXT NOT NULL,
                            payload_json TEXT NOT NULL,
                            origin TEXT NOT NULL,
                            state TEXT NOT NULL DEFAULT 'queued'
                                CHECK (state IN {str(_ALL_STATES)}),
                            priority INTEGER NOT NULL DEFAULT 0,
                            attempts INTEGER NOT NULL DEFAULT 0,
                            owner_pid INTEGER,
                            owner_started_at INTEGER,
                            claim_token TEXT,
                            resume_only INTEGER NOT NULL DEFAULT 0
                                CHECK (resume_only IN (0, 1)),
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            claimed_at REAL,
                            completed_at REAL,
                            not_before REAL NOT NULL DEFAULT 0,
                            last_error TEXT
                        );
                        CREATE INDEX IF NOT EXISTS idx_inbound_events_ready
                            ON inbound_events(state, not_before, id);
                        CREATE INDEX IF NOT EXISTS idx_inbound_events_session
                            ON inbound_events(session_key, state, id);
                        CREATE INDEX IF NOT EXISTS idx_inbound_events_owner
                            ON inbound_events(state, owner_pid, owner_started_at);
                        CREATE TABLE IF NOT EXISTS inbox_lanes (
                            session_key TEXT PRIMARY KEY,
                            last_claim_order INTEGER NOT NULL DEFAULT 0,
                            updated_at REAL NOT NULL
                        );
                        PRAGMA user_version={_SCHEMA_VERSION};
                        """
                    )
                    self._prune_rows(conn, time.time())
                    self._initialized = True
                    break
                except sqlite3.DatabaseError as exc:
                    if attempt or not _is_corruption_error(exc):
                        raise
                    if conn is not None:
                        conn.close()
                        conn = None
                    self._quarantine_database(exc)
                finally:
                    if conn is not None:
                        conn.close()
            if self._initialized:
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    logger.debug(
                        "Could not set durable inbox permissions", exc_info=True
                    )

    def _quarantine_database(self, exc: BaseException) -> Optional[Path]:
        path = self.path()
        if not path.exists():
            return None
        suffix = f".corrupt-{time.time_ns()}-{secrets.token_hex(4)}"
        quarantined = path.with_name(path.name + suffix)
        os.replace(path, quarantined)
        for sidecar_suffix in ("-wal", "-shm"):
            sidecar = Path(os.fspath(path) + sidecar_suffix)
            if sidecar.exists():
                os.replace(
                    sidecar,
                    Path(os.fspath(quarantined) + sidecar_suffix),
                )
        logger.error("Quarantined corrupt Gateway inbox at %s: %s", quarantined, exc)
        return quarantined

    @staticmethod
    def _begin(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _commit(conn: sqlite3.Connection) -> None:
        conn.execute("COMMIT")

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _prune_rows(
        self,
        conn: sqlite3.Connection,
        now: float,
        *,
        target_rows: Optional[int] = None,
    ) -> int:
        """Delete expired/overflow terminal rows and their orphaned lanes."""
        deleted = 0
        cutoff = now - self.terminal_retention_seconds
        cursor = conn.execute(
            """DELETE FROM inbound_events
               WHERE state IN ('completed','cancelled','dead_letter')
                 AND updated_at < ?""",
            (cutoff,),
        )
        deleted += max(0, int(cursor.rowcount))

        if target_rows is not None:
            total = int(
                conn.execute("SELECT COUNT(*) FROM inbound_events").fetchone()[0]
            )
            excess = max(0, total - max(0, int(target_rows)))
            if excess:
                cursor = conn.execute(
                    """DELETE FROM inbound_events WHERE id IN (
                           SELECT id FROM inbound_events
                           WHERE state IN ('completed','cancelled','dead_letter')
                           ORDER BY updated_at ASC, id ASC LIMIT ?
                       )""",
                    (excess,),
                )
                deleted += max(0, int(cursor.rowcount))

        conn.execute(
            """DELETE FROM inbox_lanes
               WHERE NOT EXISTS (
                   SELECT 1 FROM inbound_events event
                   WHERE event.session_key=inbox_lanes.session_key
               )"""
        )
        return deleted

    def _maybe_prune(self, conn: sqlite3.Connection, now: float) -> int:
        monotonic_now = time.monotonic()
        if (
            self._last_prune_monotonic
            and monotonic_now - self._last_prune_monotonic < self.prune_interval_seconds
        ):
            return 0
        deleted = self._prune_rows(conn, now)
        self._last_prune_monotonic = monotonic_now
        return deleted

    def _ensure_row_capacity(self, conn: sqlite3.Connection, now: float) -> bool:
        total = int(conn.execute("SELECT COUNT(*) FROM inbound_events").fetchone()[0])
        if total < self.max_rows:
            return True
        self._prune_rows(conn, now, target_rows=self.max_rows - 1)
        total = int(conn.execute("SELECT COUNT(*) FROM inbound_events").fetchone()[0])
        return total < self.max_rows

    def prune(self, *, now: Optional[float] = None) -> int:
        """Synchronously prune terminal history; callers should offload it."""
        prune_at = time.time() if now is None else float(now)
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                deleted = self._prune_rows(conn, prune_at, target_rows=self.max_rows)
                self._commit(conn)
                self._last_prune_monotonic = time.monotonic()
                return deleted
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> InboxRow:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InboxPayloadError(f"invalid payload JSON: {exc}") from exc
        # Validate the complete payload now, not only when the dispatcher asks
        # for it. This prevents one poisoned head row from blocking a lane.
        deserialize_message_event(payload)
        return InboxRow(
            id=int(row["id"]),
            queue_id=str(row["queue_id"]),
            dedupe_key=str(row["dedupe_key"]),
            session_key=str(row["session_key"]),
            session_id=row["session_id"],
            profile=row["profile"],
            platform=str(row["platform"]),
            trigger_identity=str(row["trigger_identity"]),
            payload=payload,
            origin=str(row["origin"]),
            state=str(row["state"]),
            priority=int(row["priority"]),
            attempts=int(row["attempts"]),
            owner_pid=row["owner_pid"],
            owner_started_at=row["owner_started_at"],
            claim_token=row["claim_token"],
            resume_only=bool(row["resume_only"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            claimed_at=row["claimed_at"],
            completed_at=row["completed_at"],
            not_before=float(row["not_before"]),
            last_error=row["last_error"],
        )

    def _quarantine_row(self, row_id: int, exc: BaseException) -> None:
        message = f"Payload quarantine: {type(exc).__name__}: {exc}"[:500]
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                conn.execute(
                    """UPDATE inbound_events
                       SET state='dead_letter', owner_pid=NULL,
                           owner_started_at=NULL, claim_token=NULL,
                           updated_at=?, completed_at=?, last_error=? WHERE id=?""",
                    (time.time(), time.time(), message, int(row_id)),
                )
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    def _decode_or_quarantine(self, row: Optional[sqlite3.Row]) -> Optional[InboxRow]:
        if row is None:
            return None
        try:
            return self._decode_row(row)
        except InboxPayloadError as exc:
            logger.error(
                "Quarantining corrupt Gateway inbox row id=%s: %s", row["id"], exc
            )
            self._quarantine_row(int(row["id"]), exc)
            return None

    def get(self, queue_id: str) -> Optional[InboxRow]:
        with self._writer_lock:
            conn = self._open()
            try:
                row = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE queue_id=?",
                    (str(queue_id),),
                ).fetchone()
            finally:
                conn.close()
        return self._decode_or_quarantine(row)

    def enqueue(
        self,
        event: Any,
        *,
        session_key: str,
        origin: str = "direct",
        priority: int = 0,
        queue_id: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        trigger_identity: Optional[str] = None,
        not_before: float = 0.0,
    ) -> EnqueueResult:
        """Commit one inbound turn, returning the existing row on dedupe."""
        session_key = str(session_key or "").strip()
        if not session_key:
            raise InboxSerializationError("session_key is required")
        origin = str(origin or "direct").strip().lower()
        if origin not in _ORIGINS:
            raise InboxSerializationError(f"unsupported inbound origin: {origin}")
        payload = serialize_message_event(event)
        payload_json = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        requested_queue_id = queue_id or _extract_explicit_queue_id(event)
        generated_queue_id = requested_queue_id is None
        queue_id = str(requested_queue_id or _new_queue_id()).strip()[:128]
        if not queue_id:
            raise InboxSerializationError("queue_id is required")
        supplied_dedupe_key = dedupe_key is not None
        dedupe_key = str(
            dedupe_key or _dedupe_key(event, session_key, queue_id)
        ).strip()
        if not dedupe_key:
            raise InboxSerializationError("dedupe_key is required")
        platform = str(payload["source"]["platform"])
        profile = payload["source"].get("profile")
        event_message_id = payload["event"].get("message_id")
        synthetic_trigger_identity = not trigger_identity and not event_message_id
        trigger_identity = str(
            trigger_identity or event_message_id or f"gateway-inbox:{queue_id}"
        )
        now = time.time()

        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                if generated_queue_id:
                    while conn.execute(
                        "SELECT 1 FROM inbound_events WHERE queue_id=?", (queue_id,)
                    ).fetchone():
                        queue_id = _new_queue_id()
                        if not supplied_dedupe_key:
                            dedupe_key = _dedupe_key(event, session_key, queue_id)
                        if synthetic_trigger_identity:
                            trigger_identity = f"gateway-inbox:{queue_id}"
                existing = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE dedupe_key=?",
                    (dedupe_key,),
                ).fetchone()
                if existing is not None:
                    self._commit(conn)
                    row = self._decode_or_quarantine(existing)
                    return EnqueueResult(True, False, "duplicate", row)

                queue_id_owner = conn.execute(
                    "SELECT 1 FROM inbound_events WHERE queue_id=?", (queue_id,)
                ).fetchone()
                if queue_id_owner is not None:
                    self._commit(conn)
                    return EnqueueResult(False, False, "queue_id_conflict")

                self._maybe_prune(conn, now)
                if not self._ensure_row_capacity(conn, now):
                    self._commit(conn)
                    return EnqueueResult(False, False, "storage_full")

                placeholders = ",".join("?" for _ in _PENDING_STATES)
                if self.max_pending_per_session:
                    per_session = conn.execute(
                        f"""SELECT COUNT(*) FROM inbound_events
                            WHERE session_key=? AND state IN ({placeholders})""",
                        (session_key, *_PENDING_STATES),
                    ).fetchone()[0]
                    if int(per_session) >= self.max_pending_per_session:
                        self._commit(conn)
                        return EnqueueResult(False, False, "session_full")
                if self.max_pending_total:
                    total = conn.execute(
                        f"SELECT COUNT(*) FROM inbound_events WHERE state IN ({placeholders})",
                        _PENDING_STATES,
                    ).fetchone()[0]
                    if int(total) >= self.max_pending_total:
                        self._commit(conn)
                        return EnqueueResult(False, False, "global_full")

                try:
                    conn.execute(
                        """INSERT INTO inbound_events (
                               queue_id, dedupe_key, session_key, profile,
                               platform, trigger_identity, payload_json, origin,
                               state, priority, created_at, updated_at, not_before
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)""",
                        (
                            queue_id,
                            dedupe_key,
                            session_key,
                            profile,
                            platform,
                            trigger_identity,
                            payload_json,
                            origin,
                            int(priority),
                            now,
                            now,
                            max(0.0, float(not_before or 0.0)),
                        ),
                    )
                except sqlite3.IntegrityError:
                    existing = conn.execute(
                        f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE dedupe_key=?",
                        (dedupe_key,),
                    ).fetchone()
                    if existing is not None:
                        self._commit(conn)
                        row = self._decode_or_quarantine(existing)
                        return EnqueueResult(True, False, "duplicate", row)
                    queue_id_owner = conn.execute(
                        "SELECT 1 FROM inbound_events WHERE queue_id=?", (queue_id,)
                    ).fetchone()
                    if queue_id_owner is not None:
                        self._commit(conn)
                        return EnqueueResult(False, False, "queue_id_conflict")
                    raise
                inserted = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE queue_id=?",
                    (queue_id,),
                ).fetchone()
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()
        return EnqueueResult(
            True, True, "enqueued", self._decode_or_quarantine(inserted)
        )

    def bind(
        self,
        queue_id: str,
        session_id: str,
        trigger_identity: Optional[str] = None,
        *,
        claim_token: Optional[str] = None,
    ) -> bool:
        """Bind final routing before claim or through an exact live claim."""
        session_id = str(session_id or "").strip()
        if not session_id:
            return False
        pid, started_at = _owner_stamp()
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                cursor = conn.execute(
                    f"""UPDATE inbound_events
                        SET session_id=?,
                            trigger_identity=COALESCE(?, trigger_identity),
                            updated_at=?
                        WHERE queue_id=? AND (
                            state IN ('queued','resume_ready')
                            OR (
                                state='claimed' AND ? IS NOT NULL
                                AND claim_token=? AND owner_pid=?
                                AND owner_started_at IS ?
                            )
                        )""",
                    (
                        session_id,
                        str(trigger_identity) if trigger_identity else None,
                        time.time(),
                        str(queue_id),
                        str(claim_token) if claim_token else None,
                        str(claim_token) if claim_token else None,
                        pid,
                        started_at,
                    ),
                )
                self._commit(conn)
                return bool(cursor.rowcount)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    @staticmethod
    def _advance_lane(conn: sqlite3.Connection, session_key: str, now: float) -> None:
        next_order = int(
            conn.execute(
                "SELECT COALESCE(MAX(last_claim_order), 0) + 1 FROM inbox_lanes"
            ).fetchone()[0]
        )
        conn.execute(
            """INSERT INTO inbox_lanes(session_key, last_claim_order, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(session_key) DO UPDATE SET
                   last_claim_order=excluded.last_claim_order,
                   updated_at=excluded.updated_at""",
            (session_key, next_order, now),
        )

    def claim(
        self,
        queue_id: str,
        *,
        claim_token: Optional[str] = None,
    ) -> Optional[InboxRow]:
        """Atomically claim one session head with a task-scoped token.

        Supplying the token returned by an earlier successful call provides
        an idempotent read for that same worker. A second token-less caller in
        the same process cannot acquire the already claimed row.
        """
        pid, started_at = _owner_stamp()
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                candidate = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE queue_id=?",
                    (str(queue_id),),
                ).fetchone()
                if candidate is None:
                    self._commit(conn)
                    return None
                if candidate["state"] == "claimed":
                    same_claim = bool(
                        claim_token
                        and secrets.compare_digest(
                            str(candidate["claim_token"] or ""), str(claim_token)
                        )
                        and candidate["owner_pid"] == pid
                        and candidate["owner_started_at"] == started_at
                    )
                    self._commit(conn)
                    return self._decode_or_quarantine(candidate) if same_claim else None
                if candidate["state"] not in {"queued", "resume_ready"}:
                    self._commit(conn)
                    return None
                if int(candidate["attempts"]) >= self.max_attempts:
                    now = time.time()
                    conn.execute(
                        """UPDATE inbound_events SET state='dead_letter',
                           owner_pid=NULL, owner_started_at=NULL, claim_token=NULL,
                           completed_at=?, updated_at=?,
                           last_error='claim attempt limit reached'
                           WHERE id=?""",
                        (now, now, candidate["id"]),
                    )
                    self._commit(conn)
                    return None
                session_key = str(candidate["session_key"])
                blocked = conn.execute(
                    """SELECT 1 FROM inbound_events
                       WHERE session_key=? AND (
                           state='claimed'
                           OR (state IN ('queued','resume_ready') AND id < ?)
                       ) LIMIT 1""",
                    (session_key, candidate["id"]),
                ).fetchone()
                if blocked is not None:
                    self._commit(conn)
                    return None
                now = time.time()
                new_token = _new_claim_token()
                cursor = conn.execute(
                    """UPDATE inbound_events SET state='claimed', attempts=attempts+1,
                       owner_pid=?, owner_started_at=?, claim_token=?,
                       claimed_at=?, updated_at=?, last_error=NULL
                       WHERE id=? AND state IN ('queued','resume_ready')""",
                    (pid, started_at, new_token, now, now, candidate["id"]),
                )
                if not cursor.rowcount:
                    self._commit(conn)
                    return None
                self._advance_lane(conn, session_key, now)
                claimed = conn.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE id=?",
                    (candidate["id"],),
                ).fetchone()
                self._commit(conn)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()
        return self._decode_or_quarantine(claimed)

    def claim_next(
        self,
        *,
        exclude_session_keys: Optional[Iterable[str]] = None,
        deliverable_platforms: Optional[Iterable[Any]] = None,
        now: Optional[float] = None,
    ) -> Optional[InboxRow]:
        """Fairly claim one ready session head.

        Sessions least recently claimed are selected first; FIFO is preserved
        within a session, and a session with an in-flight row is ineligible.
        """
        excluded = {str(item) for item in (exclude_session_keys or ())}
        platforms = None
        if deliverable_platforms is not None:
            platforms = {
                str(getattr(item, "value", item)) for item in deliverable_platforms
            }
        ready_at = time.time() if now is None else float(now)
        pid, started_at = _owner_stamp()

        # A corrupt head is quarantined and the search continues, so it cannot
        # permanently block every healthy session behind it.
        for _ in range(100):
            with self._writer_lock:
                conn = self._open()
                try:
                    self._begin(conn)
                    conn.execute(
                        """UPDATE inbound_events SET state='dead_letter',
                           owner_pid=NULL, owner_started_at=NULL, claim_token=NULL,
                           completed_at=?, updated_at=?,
                           last_error='claim attempt limit reached'
                           WHERE state IN ('queued','resume_ready') AND attempts>=?""",
                        (ready_at, ready_at, self.max_attempts),
                    )
                    where = [
                        "e.state IN ('queued','resume_ready')",
                        "e.not_before<=?",
                        "NOT EXISTS (SELECT 1 FROM inbound_events active "
                        "WHERE active.session_key=e.session_key "
                        "AND active.state='claimed')",
                        "NOT EXISTS (SELECT 1 FROM inbound_events earlier "
                        "WHERE earlier.session_key=e.session_key "
                        "AND earlier.state IN ('queued','resume_ready') "
                        "AND earlier.id<e.id)",
                    ]
                    params: list[Any] = [ready_at]
                    if excluded:
                        marks = ",".join("?" for _ in excluded)
                        where.append(f"e.session_key NOT IN ({marks})")
                        params.extend(sorted(excluded))
                    if platforms is not None:
                        if not platforms:
                            self._commit(conn)
                            return None
                        marks = ",".join("?" for _ in platforms)
                        where.append(f"e.platform IN ({marks})")
                        params.extend(sorted(platforms))
                    candidate = conn.execute(
                        f"""SELECT e.id, e.session_key
                            FROM inbound_events e
                            LEFT JOIN inbox_lanes lane
                              ON lane.session_key=e.session_key
                            WHERE {" AND ".join(where)}
                            ORDER BY COALESCE(lane.last_claim_order, 0) ASC,
                                     e.priority DESC, e.id ASC
                            LIMIT 1""",
                        params,
                    ).fetchone()
                    if candidate is None:
                        self._commit(conn)
                        return None
                    claimed_at = time.time()
                    claim_token = _new_claim_token()
                    cursor = conn.execute(
                        """UPDATE inbound_events SET state='claimed',
                           attempts=attempts+1, owner_pid=?, owner_started_at=?,
                           claim_token=?, claimed_at=?, updated_at=?, last_error=NULL
                           WHERE id=? AND state IN ('queued','resume_ready')""",
                        (
                            pid,
                            started_at,
                            claim_token,
                            claimed_at,
                            claimed_at,
                            candidate["id"],
                        ),
                    )
                    if not cursor.rowcount:
                        self._commit(conn)
                        continue
                    self._advance_lane(conn, str(candidate["session_key"]), claimed_at)
                    row = conn.execute(
                        f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE id=?",
                        (candidate["id"],),
                    ).fetchone()
                    self._commit(conn)
                except Exception:
                    self._rollback(conn)
                    raise
                finally:
                    conn.close()
            decoded = self._decode_or_quarantine(row)
            if decoded is not None:
                return decoded
        logger.error("Durable inbox encountered too many corrupt claim candidates")
        return None

    def complete(self, queue_id: str, claim_token: str) -> bool:
        """Complete only the exact task-scoped claim held by this process."""
        if not claim_token:
            return False
        now = time.time()
        pid, started_at = _owner_stamp()
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                cursor = conn.execute(
                    """UPDATE inbound_events SET state='completed',
                       owner_pid=NULL, owner_started_at=NULL, claim_token=NULL,
                       updated_at=?, completed_at=?, last_error=NULL
                       WHERE queue_id=? AND state='claimed' AND claim_token=?
                         AND owner_pid=? AND owner_started_at IS ?""",
                    (
                        now,
                        now,
                        str(queue_id),
                        str(claim_token),
                        pid,
                        started_at,
                    ),
                )
                self._maybe_prune(conn, now)
                self._commit(conn)
                return bool(cursor.rowcount)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    def retry(
        self,
        queue_id: str,
        claim_token: str,
        *,
        error: str = "worker failed",
        not_before: float = 0.0,
        resume_only: Optional[bool] = None,
    ) -> bool:
        """Release this worker's claim back to its safe dispatch state.

        A normal turn returns to ``queued``. A turn whose trigger was already
        durable returns to ``resume_ready`` so it is continued without replaying
        the user message. ``resume_only`` lets the caller supply that durable
        trigger decision for a failure in the still-live owner process; ``None``
        preserves the row's existing recovery mode. Exhausted rows are
        dead-lettered atomically.
        """
        if not claim_token:
            return False
        now = time.time()
        pid, started_at = _owner_stamp()
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                row = conn.execute(
                    """SELECT id, attempts, resume_only FROM inbound_events
                       WHERE queue_id=? AND state='claimed' AND claim_token=?
                         AND owner_pid=? AND owner_started_at IS ?""",
                    (str(queue_id), str(claim_token), pid, started_at),
                ).fetchone()
                if row is None:
                    self._commit(conn)
                    return False
                exhausted = int(row["attempts"]) >= self.max_attempts
                next_resume_only = (
                    bool(row["resume_only"])
                    if resume_only is None
                    else bool(resume_only)
                )
                target_state = (
                    "dead_letter"
                    if exhausted
                    else "resume_ready"
                    if next_resume_only
                    else "queued"
                )
                completed_at = now if exhausted else None
                retry_at = 0.0 if exhausted else max(0.0, float(not_before or 0.0))
                cursor = conn.execute(
                    """UPDATE inbound_events SET state=?, owner_pid=NULL,
                       owner_started_at=NULL, claim_token=NULL, resume_only=?,
                       updated_at=?, completed_at=?, not_before=?, last_error=?
                       WHERE id=? AND state='claimed' AND claim_token=?
                         AND owner_pid=? AND owner_started_at IS ?""",
                    (
                        target_state,
                        int(next_resume_only),
                        now,
                        completed_at,
                        retry_at,
                        (
                            "claim attempt limit reached"
                            if exhausted
                            else str(error or "worker failed")[:500]
                        ),
                        row["id"],
                        str(claim_token),
                        pid,
                        started_at,
                    ),
                )
                self._maybe_prune(conn, now)
                self._commit(conn)
                return bool(cursor.rowcount)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    def release(
        self,
        queue_id: str,
        claim_token: str,
        *,
        error: str = "worker released claim",
        not_before: float = 0.0,
        resume_only: Optional[bool] = None,
    ) -> bool:
        """Alias for :meth:`retry` for cancellation/finally paths."""
        return self.retry(
            queue_id,
            claim_token,
            error=error,
            not_before=not_before,
            resume_only=resume_only,
        )

    def cancel(self, queue_id: str, *, session_key: Optional[str] = None) -> bool:
        now = time.time()
        sql = (
            "UPDATE inbound_events SET state='cancelled', owner_pid=NULL, "
            "owner_started_at=NULL, claim_token=NULL, updated_at=?, "
            "completed_at=? WHERE queue_id=? "
            "AND state IN ('queued','resume_ready')"
        )
        params: list[Any] = [now, now, str(queue_id)]
        if session_key is not None:
            sql += " AND session_key=?"
            params.append(str(session_key))
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                cursor = conn.execute(sql, params)
                self._maybe_prune(conn, now)
                self._commit(conn)
                return bool(cursor.rowcount)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    def cancel_session(self, session_key: str, *, reason: str = "session reset") -> int:
        now = time.time()
        with self._writer_lock:
            conn = self._open()
            try:
                self._begin(conn)
                cursor = conn.execute(
                    """UPDATE inbound_events SET state='cancelled', updated_at=?,
                       owner_pid=NULL, owner_started_at=NULL, claim_token=NULL,
                       completed_at=?, last_error=?
                       WHERE session_key=?
                         AND state IN ('queued','resume_ready')""",
                    (now, now, str(reason)[:500], str(session_key)),
                )
                self._maybe_prune(conn, now)
                self._commit(conn)
                return int(cursor.rowcount)
            except Exception:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    def pending_count(self, session_key: Optional[str] = None) -> int:
        placeholders = ",".join("?" for _ in _PENDING_STATES)
        sql = f"SELECT COUNT(*) FROM inbound_events WHERE state IN ({placeholders})"
        params: list[Any] = list(_PENDING_STATES)
        if session_key is not None:
            sql += " AND session_key=?"
            params.append(str(session_key))
        with self._writer_lock:
            conn = self._open()
            try:
                return int(conn.execute(sql, params).fetchone()[0])
            finally:
                conn.close()

    def list_rows(
        self,
        *,
        session_key: Optional[str] = None,
        states: Optional[Iterable[str]] = None,
        limit: int = 1_000,
    ) -> list[InboxRow]:
        where: list[str] = []
        params: list[Any] = []
        if session_key is not None:
            where.append("session_key=?")
            params.append(str(session_key))
        if states is not None:
            normalized = [str(state) for state in states]
            if not normalized:
                return []
            marks = ",".join("?" for _ in normalized)
            where.append(f"state IN ({marks})")
            params.extend(normalized)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(max(1, int(limit)))
        with self._writer_lock:
            conn = self._open()
            try:
                rows = conn.execute(
                    f"""SELECT {_SELECT_COLUMNS} FROM inbound_events
                        {clause} ORDER BY id LIMIT ?""",
                    params,
                ).fetchall()
            finally:
                conn.close()
        decoded: list[InboxRow] = []
        for row in rows:
            item = self._decode_or_quarantine(row)
            if item is not None:
                decoded.append(item)
        return decoded

    def reclaim_dead_claims(
        self,
        trigger_is_durable: Optional[Callable[[str, str], Optional[bool]]] = None,
        *,
        deliverable_platforms: Optional[Iterable[Any]] = None,
    ) -> list[InboxRow]:
        """Reconcile claims owned by dead processes.

        ``trigger_is_durable`` returns:

        - ``True``: the user row is already in SessionDB; resume without
          replaying it.
        - ``False``: the trigger is absent; requeue the original payload.
        - ``None``: SessionDB could not be checked; leave the claim untouched
          for a later retry instead of guessing.

        Non-exhausted rows outside ``deliverable_platforms`` are left untouched.
        Returned ``queued``/``resume_ready`` rows are deliberately unowned;
        dispatchers must call :meth:`claim` or :meth:`claim_next` to obtain a
        claim token.
        """
        platforms = None
        if deliverable_platforms is not None:
            platforms = {
                str(getattr(item, "value", item)) for item in deliverable_platforms
            }

        with self._writer_lock:
            conn = self._open()
            try:
                rows = conn.execute(
                    f"""SELECT {_SELECT_COLUMNS} FROM inbound_events
                        WHERE state='claimed' ORDER BY id"""
                ).fetchall()
            finally:
                conn.close()

        recovered: list[InboxRow] = []
        for raw in rows:
            if _owner_alive(raw["owner_pid"], raw["owner_started_at"]):
                continue
            durable: Optional[bool]
            if int(raw["attempts"]) >= self.max_attempts:
                target_state = "dead_letter"
                durable = None
            else:
                if platforms is not None and str(raw["platform"]) not in platforms:
                    continue
                session_id = str(raw["session_id"] or "")
                trigger_identity = str(raw["trigger_identity"] or "")
                if not session_id or not trigger_identity:
                    durable = False
                elif trigger_is_durable is None:
                    durable = None
                else:
                    try:
                        durable = trigger_is_durable(session_id, trigger_identity)
                    except Exception:
                        logger.warning(
                            "Durable trigger check failed for inbox row %s",
                            raw["queue_id"],
                            exc_info=True,
                        )
                        durable = None
                if durable is None:
                    continue
                target_state = "resume_ready" if durable else "queued"

            now = time.time()
            if target_state == "resume_ready":
                completed_at = None
                error = "reclaimed after owner exit; trigger already durable"
                resume_only = 1
            elif target_state == "queued":
                completed_at = None
                error = "requeued after owner exit before trigger persistence"
                resume_only = 0
            else:
                completed_at = now
                error = "dead owner exceeded recovery attempt limit"
                resume_only = int(raw["resume_only"])

            with self._writer_lock:
                conn = self._open()
                try:
                    self._begin(conn)
                    cursor = conn.execute(
                        """UPDATE inbound_events SET state=?, owner_pid=NULL,
                           owner_started_at=NULL, claim_token=NULL, resume_only=?,
                           updated_at=?, completed_at=?, not_before=0, last_error=?
                           WHERE id=? AND state=? AND owner_pid IS ?
                             AND owner_started_at IS ? AND claim_token IS ?""",
                        (
                            target_state,
                            resume_only,
                            now,
                            completed_at,
                            error,
                            raw["id"],
                            raw["state"],
                            raw["owner_pid"],
                            raw["owner_started_at"],
                            raw["claim_token"],
                        ),
                    )
                    if cursor.rowcount:
                        updated = conn.execute(
                            f"SELECT {_SELECT_COLUMNS} FROM inbound_events WHERE id=?",
                            (raw["id"],),
                        ).fetchone()
                    else:
                        updated = None
                    self._maybe_prune(conn, now)
                    self._commit(conn)
                except Exception:
                    self._rollback(conn)
                    raise
                finally:
                    conn.close()
            item = self._decode_or_quarantine(updated)
            if item is not None:
                recovered.append(item)
        return recovered


__all__ = [
    "EnqueueResult",
    "EXPLICIT_QUEUE_METADATA_KEY",
    "GatewayInboxStore",
    "INBOX_METADATA_KEY",
    "InboxPayloadError",
    "InboxRow",
    "InboxSerializationError",
    "deserialize_message_event",
    "serialize_message_event",
]
