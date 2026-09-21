"""SQLite-backed storage for Idea Box: dedup, approvals, audit log.

Schema defined in architecture blueprint Appendix A.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import (
    ApprovalState,
    AuditEvent,
    AuditEventType,
    DedupResult,
    Source,
    TriageSummary,
    generate_event_id,
)


class IdeaBoxStore:
    """Thread-safe SQLite store for Idea Box data.

    Three tables:
      - ideabox_dedup: content_hash + url_fingerprint dedup
      - ideabox_approvals: approval state machine rows
      - ideabox_audit_log: per-event audit trail
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ideabox_dedup (
                    content_hash TEXT PRIMARY KEY,
                    url_fingerprint TEXT UNIQUE,
                    triage_id TEXT NOT NULL,
                    task_id TEXT,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ideabox_dedup_url
                    ON ideabox_dedup(url_fingerprint);

                CREATE TABLE IF NOT EXISTS ideabox_approvals (
                    triage_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL
                        CHECK(status IN ('pending','approved','rejected','amended')),
                    source_json TEXT NOT NULL,
                    triage_summary_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    resolved_at INTEGER,
                    resolved_by TEXT,
                    resolution_reason TEXT,
                    kanban_task_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ideabox_approvals_status
                    ON ideabox_approvals(status);
                CREATE INDEX IF NOT EXISTS idx_ideabox_approvals_task
                    ON ideabox_approvals(kanban_task_id);

                CREATE TABLE IF NOT EXISTS ideabox_audit_log (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    triage_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    actor_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ideabox_audit_triage
                    ON ideabox_audit_log(triage_id);
                CREATE INDEX IF NOT EXISTS idx_ideabox_audit_type
                    ON ideabox_audit_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_ideabox_audit_time
                    ON ideabox_audit_log(timestamp);
            """)

    # ── Dedup ──────────────────────────────────────────────────────────

    def check_dedup(self, content_hash: str, url_fingerprint: str) -> DedupResult:
        """Check if a source has been seen before."""
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT triage_id, task_id FROM ideabox_dedup "
                "WHERE content_hash = ? OR url_fingerprint = ?",
                (content_hash, url_fingerprint),
            ).fetchone()
        if row:
            return DedupResult(
                is_duplicate=True,
                existing_triage_id=row[0],
                existing_task_id=row[1],
            )
        return DedupResult()

    def record_dedup(
        self,
        content_hash: str,
        url_fingerprint: str,
        triage_id: str,
        task_id: Optional[str] = None,
    ) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ideabox_dedup "
                "(content_hash, url_fingerprint, triage_id, task_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (content_hash, url_fingerprint, triage_id, task_id, int(__import__("time").time())),
            )

    def update_dedup_task_id(self, content_hash: str, task_id: str) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE ideabox_dedup SET task_id = ? WHERE content_hash = ?",
                (task_id, content_hash),
            )

    # ── Approvals ───────────────────────────────────────────────────────

    def save_approval(self, state: ApprovalState) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ideabox_approvals "
                "(triage_id, status, source_json, triage_summary_json, "
                " created_at, resolved_at, resolved_by, resolution_reason, kanban_task_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    state.triage_id,
                    state.status,
                    _to_json(state.source),
                    _to_json(state.triage_summary),
                    state.created_at,
                    state.resolved_at,
                    state.resolved_by,
                    state.resolution_reason,
                    state.kanban_task_id,
                ),
            )

    def get_approval(self, triage_id: str) -> Optional[ApprovalState]:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM ideabox_approvals WHERE triage_id = ?",
                (triage_id,),
            ).fetchone()
        if not row:
            return None
        return ApprovalState(
            triage_id=row[0],
            status=row[1],
            source=_from_json(row[2], Source),
            triage_summary=_from_json(row[3], TriageSummary),
            created_at=row[4],
            resolved_at=row[5],
            resolved_by=row[6],
            resolution_reason=row[7],
            kanban_task_id=row[8],
        )

    def update_approval_status(
        self,
        triage_id: str,
        status: str,
        resolved_by: str,
        reason: Optional[str] = None,
        kanban_task_id: Optional[str] = None,
    ) -> None:
        now = int(__import__("time").time())
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE ideabox_approvals SET status = ?, resolved_at = ?, "
                "resolved_by = ?, resolution_reason = ?, kanban_task_id = ? "
                "WHERE triage_id = ?",
                (status, now, resolved_by, reason, kanban_task_id, triage_id),
            )

    # ── Audit Log ───────────────────────────────────────────────────────

    def log_event(self, event: AuditEvent) -> None:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO ideabox_audit_log "
                "(event_id, event_type, triage_id, timestamp, actor_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.event_type,
                    event.triage_id,
                    event.timestamp,
                    event.actor_id,
                    json.dumps(event.payload),
                ),
            )

    def get_events(self, triage_id: str, limit: int = 50) -> list[AuditEvent]:
        with self._lock, sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT event_id, event_type, triage_id, timestamp, actor_id, payload_json "
                "FROM ideabox_audit_log WHERE triage_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (triage_id, limit),
            ).fetchall()
        return [
            AuditEvent(
                event_id=r[0],
                event_type=r[1],
                triage_id=r[2],
                timestamp=r[3],
                actor_id=r[4],
                payload=json.loads(r[5]),
            )
            for r in rows
        ]


def _to_json(obj) -> str:
    """Serialize a dataclass to JSON, handling nested dataclasses."""
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(obj, default=_json_default, indent=2)
    return json.dumps(obj)


def _json_default(o):
    if hasattr(o, "__dataclass_fields__"):
        return {f: getattr(o, f) for f in o.__dataclass_fields__}
    if isinstance(o, __import__("enum").Enum):
        return o.value
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _from_json(json_str: str, cls: type) -> object:
    """Deserialize JSON to a dataclass."""
    import typing
    data = json.loads(json_str)
    if cls is Source:
        from .models import SourceType, Provenance
        prov = Provenance(**data.pop("provenance", {}))
        data["source_type"] = SourceType(data.get("source_type", "unknown"))
        return Source(provenance=prov, **data)
    if cls is TriageSummary:
        from .models import Classification, RoutingDecision, Risk, Source as Src
        src = _from_json(json.dumps(data.pop("source", {})), Src)
        cls_data = data.pop("classification", {})
        classification = Classification(**cls_data) if cls_data else Classification(category="other")
        risks = [Risk(**r) for r in data.pop("risks", [])]
        routing = RoutingDecision(**data.pop("routing", {}))
        return TriageSummary(
            source=typing.cast(Src, src),
            classification=classification,
            risks=risks,
            routing=routing,
            **data,
        )
    return data
