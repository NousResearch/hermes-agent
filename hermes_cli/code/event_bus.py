#!/usr/bin/env python3
"""Central realtime/persistent event bus for Hermes Code Mode."""

from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_cli.code.execution_policy import redact_secrets
from hermes_cli.code.github_integration import redact_github_secrets
from hermes_state import SessionDB

EVENT_VERSION = 1

_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(token|access_token|refresh_token|authorization|private_key|webhook_secret|client_secret|secret|password|HERMES_GITHUB_DEV_PAT)"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return redact_github_secrets(redact_secrets(value))
    return value


def redact_payload(value: Any, *, key: Optional[str] = None) -> Any:
    if key and _SECRET_KEY_PATTERN.search(str(key)):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_payload(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_payload(v) for v in value]
    return _redact_scalar(value)


def _normalize_type(value: str | None) -> str:
    return str(value or "code.event.unknown").strip() or "code.event.unknown"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class EventFilters:
    event_type: str | None = None
    workspace_id: str | None = None
    code_session_id: str | None = None
    orchestrated_run_id: str | None = None
    approval_id: str | None = None
    github_repo_full_name: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None = None) -> "EventFilters":
        raw = values or {}
        return cls(
            event_type=(raw.get("type") or raw.get("event_type") or None),
            workspace_id=(raw.get("workspace_id") or None),
            code_session_id=(raw.get("code_session_id") or raw.get("session_id") or None),
            orchestrated_run_id=(raw.get("orchestrated_run_id") or None),
            approval_id=(raw.get("approval_id") or None),
            github_repo_full_name=(raw.get("github_repo_full_name") or None),
        )


class CodeEventBus:
    """Central event bus: normalize, redact, persist, and fanout."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path
        self._subs_lock = threading.RLock()
        self._subscribers: dict[str, dict[str, Any]] = {}
        self._started_at = time.time()

    def _db(self) -> SessionDB:
        return SessionDB(db_path=self._db_path) if self._db_path else SessionDB()

    def _normalize_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        approval_id: str | None = None,
        github_repo_full_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": _normalize_type(event_type),
            "version": EVENT_VERSION,
            "timestamp": _now_iso(),
            "payload": redact_payload(payload or {}),
        }
        if workspace_id:
            event["workspace_id"] = str(workspace_id)
        if code_session_id:
            event["code_session_id"] = str(code_session_id)
        if orchestrated_run_id:
            event["orchestrated_run_id"] = str(orchestrated_run_id)
        if approval_id:
            event["approval_id"] = str(approval_id)
        if github_repo_full_name:
            event["github_repo_full_name"] = str(github_repo_full_name)
        if metadata:
            event["metadata"] = redact_payload(metadata)
        return event

    @staticmethod
    def _event_matches(event: dict[str, Any], filters: EventFilters) -> bool:
        if filters.event_type and str(event.get("type")) != str(filters.event_type):
            return False
        if filters.workspace_id and str(event.get("workspace_id") or "") != str(filters.workspace_id):
            return False
        if filters.code_session_id and str(event.get("code_session_id") or "") != str(filters.code_session_id):
            return False
        if filters.orchestrated_run_id and str(event.get("orchestrated_run_id") or "") != str(filters.orchestrated_run_id):
            return False
        if filters.approval_id and str(event.get("approval_id") or "") != str(filters.approval_id):
            return False
        if filters.github_repo_full_name and str(event.get("github_repo_full_name") or "") != str(filters.github_repo_full_name):
            return False
        return True

    @staticmethod
    def _inflate_row_event(row: dict[str, Any]) -> dict[str, Any]:
        raw_payload = row.get("payload_json")
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except Exception:
            payload = {}

        event = payload if isinstance(payload, dict) else {}
        event["id"] = row.get("id")
        if "type" not in event:
            event["type"] = row.get("event_type")
        if "version" not in event:
            event["version"] = EVENT_VERSION
        if "timestamp" not in event:
            event["timestamp"] = datetime.fromtimestamp(float(row.get("created_at") or time.time()), tz=timezone.utc).isoformat()
        if "workspace_id" not in event and row.get("workspace_id"):
            event["workspace_id"] = row.get("workspace_id")
        if "code_session_id" not in event and row.get("session_id"):
            event["code_session_id"] = row.get("session_id")
        if "payload" not in event:
            event["payload"] = {}
        return redact_payload(event)

    def publish(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        approval_id: str | None = None,
        github_repo_full_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "code_mode",
        level: str = "info",
    ) -> dict[str, Any]:
        event = self._normalize_event(
            event_type=event_type,
            payload=payload,
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            orchestrated_run_id=orchestrated_run_id,
            approval_id=approval_id,
            github_repo_full_name=github_repo_full_name,
            metadata=metadata,
        )

        db = self._db()
        try:
            try:
                event_id = db.append_code_event(
                    event_type=event["type"],
                    payload=event,
                    workspace_id=workspace_id,
                    code_session_id=code_session_id,
                    source=source,
                    level=level,
                )
            except sqlite3.IntegrityError:
                event.pop("workspace_id", None)
                event_id = db.append_code_event(
                    event_type=event["type"],
                    payload=event,
                    workspace_id=None,
                    code_session_id=code_session_id,
                    source=source,
                    level=level,
                )
        finally:
            db.close()
        event["id"] = event_id

        stale_ids: list[str] = []
        with self._subs_lock:
            subscribers = list(self._subscribers.items())
        for sub_id, sub in subscribers:
            filters = sub["filters"]
            if not self._event_matches(event, filters):
                continue
            try:
                sub["queue"].put_nowait(event)
            except Exception:
                stale_ids.append(sub_id)
        for sub_id in stale_ids:
            self.unsubscribe(sub_id)
        return event

    def subscribe(self, *, filters: EventFilters | dict[str, Any] | None = None, max_queue_size: int = 256) -> tuple[str, queue.Queue]:
        sub_id = str(uuid.uuid4())
        filt = filters if isinstance(filters, EventFilters) else EventFilters.from_dict(filters)
        q: queue.Queue = queue.Queue(maxsize=max(1, _safe_int(max_queue_size, 256)))
        with self._subs_lock:
            self._subscribers[sub_id] = {
                "id": sub_id,
                "filters": filt,
                "queue": q,
                "created_at": time.time(),
            }
        return sub_id, q

    def unsubscribe(self, subscription_id: str) -> None:
        with self._subs_lock:
            self._subscribers.pop(subscription_id, None)

    def _find_rowid_by_event_id(self, event_id: str) -> Optional[int]:
        if not event_id:
            return None
        db = self._db()
        try:
            with db._lock:
                row = db._conn.execute(
                    "SELECT rowid AS rowid FROM code_events WHERE id = ? LIMIT 1",
                    (event_id,),
                ).fetchone()
            if not row:
                return None
            return int(row["rowid"] if hasattr(row, "keys") else row[0])
        finally:
            db.close()

    def fetch_events(
        self,
        *,
        filters: EventFilters | dict[str, Any] | None = None,
        since_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        filt = filters if isinstance(filters, EventFilters) else EventFilters.from_dict(filters)
        max_limit = max(1, min(_safe_int(limit, 50), 1000))
        safe_offset = max(0, _safe_int(offset, 0))

        where = []
        params: list[Any] = []
        if filt.workspace_id:
            where.append("workspace_id = ?")
            params.append(filt.workspace_id)
        if filt.code_session_id:
            where.append("session_id = ?")
            params.append(filt.code_session_id)
        if filt.event_type:
            where.append("event_type = ?")
            params.append(filt.event_type)

        since_rowid = self._find_rowid_by_event_id(str(since_id or ""))
        if since_rowid is not None:
            where.append("rowid > ?")
            params.append(int(since_rowid))

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        order_sql = "ORDER BY rowid DESC" if newest_first else "ORDER BY rowid ASC"

        db = self._db()
        try:
            with db._lock:
                rows = db._conn.execute(
                    f"""
                    SELECT rowid AS _rowid, id, workspace_id, session_id, event_type, payload_json, created_at
                    FROM code_events
                    {where_sql}
                    {order_sql}
                    LIMIT ? OFFSET ?
                    """,
                    tuple(params + [max_limit, safe_offset]),
                ).fetchall()
            events = [self._inflate_row_event(dict(row)) for row in rows]
        finally:
            db.close()

        filtered = [event for event in events if self._event_matches(event, filt)]
        return filtered

    def replay(
        self,
        *,
        filters: EventFilters | dict[str, Any] | None = None,
        since_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        # replay/catch-up uses persisted insertion order (rowid ASC).
        return self.fetch_events(
            filters=filters,
            since_id=since_id,
            limit=max(1, min(_safe_int(limit, 100), 2000)),
            offset=0,
            newest_first=False,
        )

    def summary(
        self,
        *,
        filters: EventFilters | dict[str, Any] | None = None,
        recent_limit: int = 200,
    ) -> dict[str, Any]:
        events = self.fetch_events(
            filters=filters,
            since_id=None,
            limit=max(1, min(_safe_int(recent_limit, 200), 2000)),
            offset=0,
            newest_first=True,
        )
        by_type: dict[str, int] = {}
        for event in events:
            et = str(event.get("type") or "code.event.unknown")
            by_type[et] = by_type.get(et, 0) + 1
        return {
            "total": len(events),
            "by_type": by_type,
            "latest_event_id": events[0]["id"] if events else None,
            "latest_timestamp": events[0]["timestamp"] if events else None,
        }

    def subscription_stats(self) -> dict[str, Any]:
        with self._subs_lock:
            subscribers = list(self._subscribers.values())
        filters_summary: list[dict[str, Any]] = []
        for sub in subscribers:
            filt: EventFilters = sub["filters"]
            filters_summary.append(
                {
                    "type": filt.event_type,
                    "workspace_id": filt.workspace_id,
                    "code_session_id": filt.code_session_id,
                    "orchestrated_run_id": filt.orchestrated_run_id,
                    "approval_id": filt.approval_id,
                    "github_repo_full_name": filt.github_repo_full_name,
                }
            )
        return {
            "active_subscribers": len(subscribers),
            "uptime_seconds": max(0, int(time.time() - self._started_at)),
            "filters": filters_summary,
        }


_BUS_LOCK = threading.RLock()
_DEFAULT_BUS: CodeEventBus | None = None


def get_code_event_bus(db_path: Optional[Path] = None) -> CodeEventBus:
    global _DEFAULT_BUS
    if db_path is not None:
        return CodeEventBus(db_path=db_path)
    with _BUS_LOCK:
        if _DEFAULT_BUS is None:
            _DEFAULT_BUS = CodeEventBus()
        return _DEFAULT_BUS


def build_event_filters_from_query(
    *,
    event_type: str | None = None,
    workspace_id: str | None = None,
    code_session_id: str | None = None,
    orchestrated_run_id: str | None = None,
    approval_id: str | None = None,
    github_repo_full_name: str | None = None,
) -> EventFilters:
    return EventFilters(
        event_type=event_type,
        workspace_id=workspace_id,
        code_session_id=code_session_id,
        orchestrated_run_id=orchestrated_run_id,
        approval_id=approval_id,
        github_repo_full_name=github_repo_full_name,
    )


__all__ = [
    "CodeEventBus",
    "EventFilters",
    "build_event_filters_from_query",
    "get_code_event_bus",
    "redact_payload",
    "_parse_bool",
]
