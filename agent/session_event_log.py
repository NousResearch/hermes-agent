"""Event Log — append-only structured event store for Hermes 2.8.

Records Task lifecycle events so every state change is traceable. Uses a
separate SQLite database (events.db) — never mixed with state.db.

Sprint 1 event types (minimal — lifecycle traceability only):
  task_created, task_updated, status_changed,
  execution_started, execution_completed, execution_failed,
  artifact_created

Design:
- Append-only: no UPDATE or DELETE in normal operation.
- Each event has a fixed payload schema per type; listed fields must exist.
- events.db is independent of state.db — no cross-db queries in Sprint 1.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── Event types (Sprint 1) ──
EVENT_TASK_CREATED = "task_created"
EVENT_TASK_UPDATED = "task_updated"
EVENT_STATUS_CHANGED = "status_changed"
EVENT_EXECUTION_STARTED = "execution_started"
EVENT_EXECUTION_COMPLETED = "execution_completed"
EVENT_EXECUTION_FAILED = "execution_failed"
EVENT_ARTIFACT_CREATED = "artifact_created"

SPRINT_1_EVENT_TYPES = frozenset([
    EVENT_TASK_CREATED,
    EVENT_TASK_UPDATED,
    EVENT_STATUS_CHANGED,
    EVENT_EXECUTION_STARTED,
    EVENT_EXECUTION_COMPLETED,
    EVENT_EXECUTION_FAILED,
    EVENT_ARTIFACT_CREATED,
])

# ── Required payload keys per event type ──
REQUIRED_PAYLOAD_KEYS: Dict[str, List[str]] = {
    EVENT_TASK_CREATED: ["task_category", "execution_mode"],
    EVENT_STATUS_CHANGED: ["from_status", "to_status", "actor"],
    EVENT_EXECUTION_COMPLETED: ["result_type"],
    EVENT_EXECUTION_FAILED: ["error_type", "error_message", "retryable"],
    EVENT_ARTIFACT_CREATED: ["artifact_type", "artifact_path"],
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


@dataclass
class SessionEvent:
    event_id: str
    session_id: str
    task_id: str
    type: str
    timestamp: float
    source: str  # "model" | "user" | "system"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "source": self.source,
            "payload": self.payload,
        }

    def validate(self) -> List[str]:
        """Check required payload keys. Returns list of missing key names."""
        required = REQUIRED_PAYLOAD_KEYS.get(self.type, [])
        return [k for k in required if k not in self.payload]


# ── SQLite helpers ──

def _get_db_path() -> Path:
    return get_hermes_home() / "events.db"


class EventLog:
    """Append-only event store backed by SQLite events.db.

    Thread-safe write via a local lock (not WAL-dependent).  All writes are
    immediate INSERT — no buffering, no batching.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path else _get_db_path()
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ── connection management ──

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._conn = sqlite3.connect(str(self._db_path))
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                    self._conn.executescript(SCHEMA_SQL)
                    self._conn.commit()
        return self._conn

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── write ──

    def write_event(self, event: SessionEvent) -> None:
        missing = event.validate()
        if missing:
            raise ValueError(
                f"Event type '{event.type}' missing required payload keys: {missing}"
            )
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO events (event_id, session_id, task_id, type, timestamp, source, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.session_id,
                    event.task_id,
                    event.type,
                    event.timestamp,
                    event.source,
                    json.dumps(event.payload, ensure_ascii=False),
                ),
            )
            conn.commit()

    # ── convenience factories ──

    def log_task_created(
        self,
        task_id: str,
        session_id: str,
        task_category: str,
        execution_mode: str,
        raw_user_request_preview: str = "",
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_TASK_CREATED,
            timestamp=time.time(),
            source="system",
            payload={
                "task_category": task_category,
                "execution_mode": execution_mode,
                "raw_user_request_preview": raw_user_request_preview[:200],
            },
        )
        self.write_event(event)
        return event

    def log_status_changed(
        self,
        task_id: str,
        session_id: str,
        from_status: str,
        to_status: str,
        reason: str = "",
        actor: str = "main_agent",
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_STATUS_CHANGED,
            timestamp=time.time(),
            source="system",
            payload={
                "from_status": from_status,
                "to_status": to_status,
                "reason": reason,
                "actor": actor,
            },
        )
        self.write_event(event)
        return event

    def log_execution_started(
        self,
        task_id: str,
        session_id: str,
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_EXECUTION_STARTED,
            timestamp=time.time(),
            source="system",
            payload={},
        )
        self.write_event(event)
        return event

    def log_execution_completed(
        self,
        task_id: str,
        session_id: str,
        result_type: str = "text",
        artifact_count: int = 0,
        turn_count: int = 0,
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_EXECUTION_COMPLETED,
            timestamp=time.time(),
            source="system",
            payload={
                "result_type": result_type,
                "artifact_count": artifact_count,
                "turn_count": turn_count,
            },
        )
        self.write_event(event)
        return event

    def log_execution_failed(
        self,
        task_id: str,
        session_id: str,
        error_type: str,
        error_message: str,
        retryable: bool = False,
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_EXECUTION_FAILED,
            timestamp=time.time(),
            source="system",
            payload={
                "error_type": error_type,
                "error_message": error_message[:500],
                "retryable": retryable,
            },
        )
        self.write_event(event)
        return event

    def log_artifact_created(
        self,
        task_id: str,
        session_id: str,
        artifact_type: str,
        artifact_path: str,
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_ARTIFACT_CREATED,
            timestamp=time.time(),
            source="system",
            payload={
                "artifact_type": artifact_type,
                "artifact_path": artifact_path,
            },
        )
        self.write_event(event)
        return event

    def log_task_updated(
        self,
        task_id: str,
        session_id: str,
        updated_fields: List[str],
    ) -> SessionEvent:
        event = SessionEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            task_id=task_id,
            type=EVENT_TASK_UPDATED,
            timestamp=time.time(),
            source="system",
            payload={"updated_fields": updated_fields},
        )
        self.write_event(event)
        return event

    # ── read (for verification / debugging) ──

    def get_events_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT event_id, session_id, task_id, type, timestamp, source, payload_json "
            "FROM events WHERE task_id = ? ORDER BY timestamp",
            (task_id,),
        ).fetchall()
        return [
            {
                "event_id": r[0],
                "session_id": r[1],
                "task_id": r[2],
                "type": r[3],
                "timestamp": r[4],
                "source": r[5],
                "payload": json.loads(r[6]),
            }
            for r in rows
        ]

    def get_events_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT event_id, session_id, task_id, type, timestamp, source, payload_json "
            "FROM events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [
            {
                "event_id": r[0],
                "session_id": r[1],
                "task_id": r[2],
                "type": r[3],
                "timestamp": r[4],
                "source": r[5],
                "payload": json.loads(r[6]),
            }
            for r in rows
        ]
