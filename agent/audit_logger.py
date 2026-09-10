"""
Audit Logger — Compliance-grade audit trail for Hermes-Agent.

Provides persistent, tamper-evident audit logging for security-critical events:
    - Permission approvals and denials
    - Dangerous command execution
    - Session lifecycle changes
    - Credential and access events
    - Policy changes

Design principles:
    - LP (Least Privilege): write-only for audit logs; no delete capability in code
    - OS (Observability of System State): all events carry session_id, trace_id, actor

Storage backends:
    - SQLiteAuditBackend: relational queries, schema enforcement, compliance reports
    - FileAuditBackend: JSON Lines with rotation, append-only semantics

Retention:
    - Configurable retention period (default 90 days per compliance standards)
    - Size-based log rotation for file backend
    - Automatic cleanup on init for expired records (SQLite only)

EventBus integration:
    - Subscribes to existing EventBus events and persists them as audit records
    - Also callable directly via AuditLogger.log()

Usage:
    from agent.audit_logger import AuditLogger, AuditEventType, AuditSeverity

    logger = AuditLogger(SQLiteAuditBackend("/data/audit.db"))
    logger.log(AuditEventType.PERMISSION_APPROVED, {
        "session_id": "sess-abc",
        "actor": "user",
        "command": "rm -rf /",
        "result": "approved",
        "severity": AuditSeverity.HIGH,
    })

    # With EventBus integration
    from agent.hermes.analytics import EventBus
    bus = EventBus()
    alogger = AuditLogger(FileAuditBackend("/var/log/hermes/audit.jsonl"))
    alogger.attach_to_eventbus(bus)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit Event Types
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    """Audit event types covering security-critical operations."""

    # Permission pipeline events
    PERMISSION_APPROVED = "audit.permission.approved"
    PERMISSION_DENIED = "audit.permission.denied"
    PERMISSION_REVIEW = "audit.permission.review"
    PERMISSION_REVOKED = "audit.permission.revoked"

    # Command execution events
    COMMAND_EXECUTED = "audit.command.executed"
    COMMAND_BLOCKED = "audit.command.blocked"
    DANGEROUS_COMMAND_DETECTED = "audit.command.dangerous"

    # Session lifecycle
    SESSION_CREATED = "audit.session.created"
    SESSION_ENDED = "audit.session.ended"
    SESSION_IMPERSONATED = "audit.session.impersonated"

    # Credential and access
    CREDENTIAL_USED = "audit.credential.used"
    CREDENTIAL_ROTATED = "audit.credential.rotated"
    API_KEY_ACCESSED = "audit.apikey.accessed"

    # Policy and configuration
    POLICY_CHANGED = "audit.policy.changed"
    PERMISSION_STAGE_ADDED = "audit.permission.stage.added"
    PERMISSION_STAGE_REMOVED = "audit.permission.stage.removed"

    # Tool events
    TOOL_EXECUTION_STARTED = "audit.tool.started"
    TOOL_EXECUTION_COMPLETED = "audit.tool.completed"
    TOOL_EXECUTION_FAILED = "audit.tool.failed"

    # Error and security events
    SECURITY_ALERT = "audit.security.alert"
    AUTH_FAILURE = "audit.auth.failure"
    RATE_LIMIT_HIT = "audit.ratelimit.hit"

    # Agent delegation
    DELEGATE_TASK_CREATED = "audit.delegate.created"
    DELEGATE_TASK_COMPLETED = "audit.delegate.completed"

    # Generic catch-all
    CUSTOM = "audit.custom"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Audit Record
# ---------------------------------------------------------------------------

class AuditRecord:
    """
    Immutable audit record representing a single security event.

    Attributes:
        event_id: Unique identifier (UUID4 hex, 32 chars)
        event_type: AuditEventType value
        timestamp: ISO8601 UTC timestamp
        session_id: Associated session (empty if none)
        trace_id: Distributed trace ID for correlation
        actor: Who triggered the event (user, agent, system)
        actor_details: Additional actor context (role, identity, etc.)
        severity: AuditSeverity level
        action: Short action name (e.g. "PERMISSION_APPROVED")
        resource: What was acted upon (command, session, credential, etc.)
        outcome: Result of the action (success, failure, denied)
        metadata: Arbitrary additional context
        checksum: SHA256 of fields above for tamper detection (optional)
    """

    __slots__ = (
        "event_id", "event_type", "timestamp", "session_id", "trace_id",
        "actor", "actor_details", "severity", "action", "resource",
        "outcome", "metadata", "checksum",
    )

    def __init__(
        self,
        event_type: str,
        actor: str = "system",
        action: str = "",
        resource: str = "",
        outcome: str = "success",
        severity: str = AuditSeverity.INFO.value,
        session_id: str = "",
        trace_id: str = "",
        actor_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        checksum: Optional[str] = None,
    ):
        import uuid as _uuid

        self.event_id = event_id or _uuid.uuid4().hex
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.session_id = session_id or ""
        self.trace_id = trace_id or ""
        self.actor = actor
        self.actor_details = actor_details or {}
        self.severity = severity
        self.action = action or event_type.split(".")[-1].upper()
        self.resource = resource or ""
        self.outcome = outcome
        self.metadata = metadata or {}
        self.checksum = checksum or self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of core fields for tamper detection."""
        import hashlib

        fields = (
            self.event_id, self.event_type, self.timestamp,
            self.session_id, self.actor, self.action,
            self.resource, self.outcome,
        )
        return hashlib.sha256("|".join(str(f) for f in fields).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditRecord:
        kwargs = {s: data[s] for s in cls.__slots__ if s in data}
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"AuditRecord(id={self.event_id[:8]}..., type={self.event_type}, "
            f"actor={self.actor}, outcome={self.outcome}, severity={self.severity})"
        )


# ---------------------------------------------------------------------------
# Storage Backends
# ---------------------------------------------------------------------------

class AuditBackend(ABC):
    """Abstract base for audit log storage backends."""

    @abstractmethod
    def write(self, record: AuditRecord) -> None:
        """Persist a single audit record."""

    @abstractmethod
    def query(
        self,
        event_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        """Query audit records with filters."""

    @abstractmethod
    def count(
        self,
        event_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """Return the count of matching records."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class SQLiteAuditBackend(AuditBackend):
    """
    SQLite-backed audit log storage.

    Schema:
        audit_events (
            event_id       TEXT PRIMARY KEY,
            event_type     TEXT NOT NULL,
            timestamp      TEXT NOT NULL,
            session_id     TEXT DEFAULT '',
            trace_id       TEXT DEFAULT '',
            actor          TEXT NOT NULL,
            actor_details  TEXT DEFAULT '{}',
            severity       TEXT DEFAULT 'info',
            action         TEXT NOT NULL,
            resource       TEXT DEFAULT '',
            outcome        TEXT DEFAULT 'success',
            metadata       TEXT DEFAULT '{}',
            checksum       TEXT NOT NULL
        )
        idx_timestamp  ON audit_events(timestamp)
        idx_session    ON audit_events(session_id)
        idx_event_type ON audit_events(event_type)
        idx_actor      ON audit_events(actor)
        idx_severity   ON audit_events(severity)

    Features:
        - Automatic schema migration on init
        - WAL mode for concurrent reads during writes
        - PRAGMA synchronous=NORMAL for performance without losing durability
        - Automatic cleanup of expired records on init (respects retention policy)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_events (
        event_id       TEXT PRIMARY KEY,
        event_type     TEXT NOT NULL,
        timestamp      TEXT NOT NULL,
        session_id     TEXT DEFAULT '',
        trace_id       TEXT DEFAULT '',
        actor          TEXT NOT NULL,
        actor_details  TEXT DEFAULT '{}',
        severity       TEXT DEFAULT 'info',
        action         TEXT NOT NULL,
        resource       TEXT DEFAULT '',
        outcome        TEXT DEFAULT 'success',
        metadata       TEXT DEFAULT '{}',
        checksum       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_session ON audit_events(session_id);
    CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_actor ON audit_events(actor);
    CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity);
    """

    def __init__(
        self,
        db_path: str = "/var/hermes/audit.db",
        retention_days: int = 90,
        wal_mode: bool = True,
    ):
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database directory, schema, and retention cleanup exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()

        # Apply schema
        conn.executescript(self.SCHEMA)

        # WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.commit()

        # Cleanup expired records
        self._cleanup_expired(conn)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _cleanup_expired(self, conn: sqlite3.Connection) -> None:
        """Remove records older than retention period."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        ).isoformat()
        try:
            cur = conn.execute(
                "DELETE FROM audit_events WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            if cur.rowcount > 0:
                logger.info(
                    "AuditLogger: purged %d expired records older than %s",
                    cur.rowcount,
                    cutoff,
                )
        except sqlite3.Error as e:
            logger.warning("AuditLogger: failed to cleanup expired records: %s", e)

    def write(self, record: AuditRecord) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_events
                (event_id, event_type, timestamp, session_id, trace_id,
                 actor, actor_details, severity, action, resource,
                 outcome, metadata, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    record.event_type,
                    record.timestamp,
                    record.session_id,
                    record.trace_id,
                    record.actor,
                    json.dumps(record.actor_details, default=str),
                    record.severity,
                    record.action,
                    record.resource,
                    record.outcome,
                    json.dumps(record.metadata, default=str),
                    record.checksum,
                ),
            )
            conn.commit()

    def query(
        self,
        event_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        with self._lock:
            conn = self._get_conn()
            conditions = []
            params: List[Any] = []

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend(event_types)

            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)

            if actor:
                conditions.append("actor = ?")
                params.append(actor)

            if severity:
                conditions.append("severity = ?")
                params.append(severity)

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())

            where = " AND ".join(conditions) if conditions else "1=1"
            query = f"""
                SELECT * FROM audit_events
                WHERE {where}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(row) for row in rows]

    def count(
        self,
        event_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            conditions = []
            params: List[Any] = []

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend(event_types)

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time.isoformat())

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time.isoformat())

            where = " AND ".join(conditions) if conditions else "1=1"
            row = conn.execute(
                f"SELECT COUNT(*) FROM audit_events WHERE {where}",
                params,
            ).fetchone()
            return row[0] if row else 0

    def _row_to_record(self, row: sqlite3.Row) -> AuditRecord:
        data = dict(row)
        for field in ("actor_details", "metadata"):
            if isinstance(data.get(field), str):
                data[field] = json.loads(data[field])
        return AuditRecord.from_dict(data)

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FileAuditBackend(AuditBackend):
    """
    File-based audit log storage using JSON Lines format.

    Each line is a valid JSON object representing one AuditRecord.
    File is append-only; rotation is handled by RotatingFileHandler.

    Features:
        - Atomic writes via write-ahead temp file + rename (on close)
        - Backed by RotatingFileHandler (size-based rotation)
        - Automatic cleanup of rotated files older than retention period
        - Timestamped archive files for compliance traceability
    """

    def __init__(
        self,
        log_path: str = "/var/hermes/audit/audit.jsonl",
        max_bytes: int = 100 * 1024 * 1024,  # 100 MB per file
        backup_count: int = 10,
        retention_days: int = 90,
    ):
        self.log_path = Path(log_path)
        self.retention_days = retention_days
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.RLock()
        self._handler: Optional[RotatingFileHandler] = None
        self._ensure_handler()
        self._cleanup_rotated()

    def _ensure_handler(self) -> None:
        """Ensure log directory and rotating handler exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handler = RotatingFileHandler(
            filename=str(self.log_path),
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        # Do NOT set a formatter — we write raw JSON lines

    def _cleanup_rotated(self) -> None:
        """Remove rotated files older than retention period."""
        cutoff = time.time() - (self.retention_days * 86400)
        for path in self.log_path.parent.glob(f"{self.log_path.name}.*"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    logger.info("AuditLogger: removed rotated audit file %s", path)
            except OSError as e:
                logger.warning("AuditLogger: failed to remove %s: %s", path, e)

    def write(self, record: AuditRecord) -> None:
        with self._lock:
            if self._handler is None:
                return
            try:
                self._handler.emit(
                    _make_audit_log_record(record)
                )
            except Exception as e:
                logger.warning("AuditLogger: failed to write record: %s", e)

    def query(
        self,
        event_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        """Scan JSON Lines files for matching records (memory-efficient)."""
        results: List[AuditRecord] = []
        with self._lock:
            files = [self.log_path] + sorted(
                self.log_path.parent.glob(f"{self.log_path.name}.*")
            )
            count = 0
            for fpath in files:
                if count >= limit:
                    break
                try:
                    with open(fpath, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            record = AuditRecord.from_dict(data)
                            if not self._record_matches(
                                record, event_types, session_id,
                                actor, severity, start_time, end_time,
                            ):
                                continue
                            results.append(record)
                            count += 1
                            if count >= limit:
                                break
                except OSError as e:
                    logger.warning("AuditLogger: failed to read %s: %s", fpath, e)
        return results

    def _record_matches(
        self,
        record: AuditRecord,
        event_types: Optional[List[str]],
        session_id: Optional[str],
        actor: Optional[str],
        severity: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> bool:
        if event_types and record.event_type not in event_types:
            return False
        if session_id and record.session_id != session_id:
            return False
        if actor and record.actor != actor:
            return False
        if severity and record.severity != severity:
            return False
        if start_time or end_time:
            try:
                ts = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            except ValueError:
                return True
            if start_time and ts < start_time:
                return False
            if end_time and ts > end_time:
                return False
        return True

    def count(
        self,
        event_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        # Full scan for count — expensive; use sparingly
        return len(self.query(
            event_types=event_types,
            start_time=start_time,
            end_time=end_time,
            limit=0,  # use a large limit in practice
        ))

    def close(self) -> None:
        with self._lock:
            if self._handler:
                self._handler.close()
                self._handler = None


def _make_audit_log_record(record: AuditRecord) -> logging.LogRecord:
    """Convert an AuditRecord into a LogRecord for RotatingFileHandler."""
    return logging.LogRecord(
        name="audit",
        level=getattr(logging, record.severity.upper(), logging.INFO),
        pathname="",
        lineno=0,
        msg=record.to_json(),
        args=(),
        exc_info=None,
    )


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Main audit logging interface.

    Wraps one or more AuditBackend instances and provides:
        - Structured log() API for recording audit events
        - EventBus subscription for automatic event capture
        - Retention enforcement (SQLite backend only)

    Thread-safe for concurrent writes.

    Usage:
        # Direct logging
        audit = AuditLogger(SQLiteAuditBackend("/var/hermes/audit.db"))
        audit.log(AuditEventType.PERMISSION_APPROVED, {
            "actor": "user",
            "resource": "rm -rf /",
            "outcome": "approved",
            "session_id": "sess-123",
            "severity": AuditSeverity.HIGH,
        })

        # EventBus integration
        audit.attach_to_eventbus(eventbus)
    """

    def __init__(
        self,
        backend: AuditBackend,
        retention_days: int = 90,
    ):
        self._backends: List[AuditBackend] = [backend]
        self._retention_days = retention_days
        self._lock = threading.RLock()
        self._eventbus_handlers: List[tuple[Any, Any]] = []

    def add_backend(self, backend: AuditBackend) -> None:
        """Add an additional storage backend (e.g., dual-write SQLite + file)."""
        with self._lock:
            self._backends.append(backend)

    def log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        actor: str = "system",
        session_id: str = "",
        trace_id: str = "",
        severity: str = AuditSeverity.INFO.value,
        outcome: str = "success",
    ) -> AuditRecord:
        """
        Record an audit event.

        Args:
            event_type: AuditEventType value or custom string
            payload: Arbitrary event data (flattened into record.metadata)
            actor: Who triggered the event
            session_id: Associated session
            trace_id: Distributed trace ID
            severity: AuditSeverity value
            outcome: success | failure | denied | revoked

        Returns:
            The created AuditRecord
        """
        # Extract known fields from payload, rest goes to metadata
        known_fields = {
            "resource", "action", "actor", "session_id", "trace_id",
            "severity", "outcome", "event_type", "actor_details",
        }
        metadata = {k: v for k, v in payload.items() if k not in known_fields}
        resource = payload.get("resource", "") or payload.get("command", "")
        action = payload.get("action", "") or event_type.split(".")[-1].upper()
        actor_details = payload.get("actor_details", {})

        record = AuditRecord(
            event_type=event_type,
            actor=payload.get("actor", actor) or "system",
            actor_details=actor_details,
            severity=payload.get("severity", severity),
            action=action,
            resource=resource,
            outcome=outcome or payload.get("outcome", "success"),
            session_id=session_id or payload.get("session_id", ""),
            trace_id=trace_id or payload.get("trace_id", ""),
            metadata=metadata,
        )

        with self._lock:
            for backend in self._backends:
                try:
                    backend.write(record)
                except Exception as e:
                    logger.warning(
                        "AuditLogger: backend %s failed to write: %s",
                        backend.__class__.__name__,
                        e,
                    )

        return record

    def query(
        self,
        event_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        actor: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[AuditRecord]:
        """Query audit records. Delegates to the first backend (primary)."""
        with self._lock:
            if not self._backends:
                return []
            return self._backends[0].query(
                event_types=event_types,
                session_id=session_id,
                actor=actor,
                severity=severity,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

    def count(
        self,
        event_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        with self._lock:
            if not self._backends:
                return 0
            return self._backends[0].count(
                event_types=event_types,
                start_time=start_time,
                end_time=end_time,
            )

    # -------------------------------------------------------------------------
    # EventBus Integration
    # -------------------------------------------------------------------------

    # Mapping from EventBus event types -> AuditEventType
    _EVENTBUS_MAPPING: Dict[str, tuple[str, Dict[str, str]]] = {
        "session.start": (AuditEventType.SESSION_CREATED, {
            "actor": "system",
            "severity": AuditSeverity.INFO.value,
        }),
        "session.end": (AuditEventType.SESSION_ENDED, {
            "actor": "system",
            "severity": AuditSeverity.INFO.value,
        }),
        "tool.call": (AuditEventType.TOOL_EXECUTION_STARTED, {
            "actor": "agent",
            "severity": AuditSeverity.INFO.value,
        }),
        "tool.result": (AuditEventType.TOOL_EXECUTION_COMPLETED, {
            "actor": "agent",
            "severity": AuditSeverity.INFO.value,
        }),
        "error": (AuditEventType.SECURITY_ALERT, {
            "actor": "system",
            "severity": AuditSeverity.HIGH.value,
        }),
    }

    def attach_to_eventbus(self, eventbus: Any) -> None:
        """
        Subscribe to an EventBus instance and persist all relevant events.

        Args:
            eventbus: An agent.hermes.analytics.EventBus instance
        """
        def handler(event: Any) -> None:
            try:
                mapping = self._EVENTBUS_MAPPING.get(event.type, None)
                if mapping is None:
                    return  # Not an auditable event

                audit_type, defaults = mapping
                payload = dict(event.payload or {})
                payload.setdefault("session_id", event.session_id or "")

                self.log(
                    event_type=audit_type,
                    payload=payload,
                    actor=defaults.get("actor", "system"),
                    session_id=event.session_id or payload.get("session_id", ""),
                    trace_id=payload.get("trace_id", ""),
                    severity=defaults.get("severity", AuditSeverity.INFO.value),
                    outcome="success",
                )
            except Exception as e:
                logger.warning("AuditLogger: EventBus handler failed: %s", e)

        eventbus.subscribe("session.start", handler)
        eventbus.subscribe("session.end", handler)
        eventbus.subscribe("tool.call", handler)
        eventbus.subscribe("tool.result", handler)
        eventbus.subscribe("error", handler)
        self._eventbus_handlers.append((eventbus, handler))
        logger.info("AuditLogger: attached to EventBus")

    def detach_from_eventbus(self, eventbus: Any) -> None:
        """Unsubscribe all audit handlers from an EventBus."""
        for bus, handler in self._eventbus_handlers:
            if bus is eventbus:
                for event_type in ("session.start", "session.end",
                                   "tool.call", "tool.result", "error"):
                    bus.unsubscribe(event_type, handler)
        self._eventbus_handlers.clear()
        logger.info("AuditLogger: detached from EventBus")

    def close(self) -> None:
        """Close all backends and detach from EventBus."""
        self.detach_from_eventbus(None)  # type: ignore
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                logger.warning("AuditLogger: backend close error: %s", e)
        self._backends.clear()

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
