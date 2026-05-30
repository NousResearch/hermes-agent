"""Durable product error/crash events for the Dev back gate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


PRODUCT_EVENT_TYPES = {
    "product.crash",
    "product.unclean_shutdown",
    "product.uncaught_exception",
    "product.api_failure",
    "product.flow_failed",
}
DEFAULT_BATCH_LIMIT = 50
MAX_MESSAGE_CHARS = 240
MAX_CONTEXT_KEYS = 24
MAX_CONTEXT_VALUE_CHARS = 180

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_product_events (
    event_id TEXT PRIMARY KEY,
    received_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    client_ts REAL,
    type TEXT NOT NULL,
    app_version TEXT,
    session_id TEXT,
    signature TEXT NOT NULL UNIQUE,
    message_redacted TEXT,
    context TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_dev_product_events_received_at
    ON dev_product_events(received_at DESC);

CREATE INDEX IF NOT EXISTS idx_dev_product_events_type
    ON dev_product_events(type, received_at DESC);
"""

SECRET_PATTERNS = [
    re.compile(r"(?i)\b(bearer|token|api[_-]?key|authorization|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?:file://)?/(?:Users|private|var|tmp|Volumes)/[^\s,;:)]+"),
    re.compile(r"\b[A-Za-z0-9_-]{40,}\b"),
]


class DevProductEventStore:
    """SQLite store that aggregates repeated product events by stable signature."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        self._lock = threading.Lock()
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def ingest_batch(self, events: Any, *, batch_limit: Optional[int] = None) -> Dict[str, Any]:
        if isinstance(events, dict):
            events = events.get("events") or []
        if not isinstance(events, list):
            raise ValueError("Product event ingest expects an events array.")
        limit = max(1, min(int(batch_limit or _env_int("HERMES_PRODUCT_EVENTS_BATCH_LIMIT", DEFAULT_BATCH_LIMIT)), 200))
        accepted = 0
        rejected: list[Dict[str, Any]] = []
        stored: list[Dict[str, Any]] = []
        for index, raw in enumerate(events[:limit]):
            try:
                normalized = normalize_product_event(raw)
                stored.append(self.upsert_event(normalized))
                accepted += 1
            except Exception as exc:
                rejected.append({"index": index, "reason": str(exc)})
        overflow = max(len(events) - limit, 0)
        if overflow:
            rejected.append({"index": limit, "reason": f"batch limit exceeded; {overflow} event(s) ignored"})
        return {
            "ok": True,
            "object": "hermes.dev_product_events_ingest",
            "accepted": accepted,
            "rejected": len(rejected),
            "rejections": rejected[:20],
            "events": stored,
        }

    def upsert_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM dev_product_events WHERE signature = ?",
                (event["signature"],),
            ).fetchone()
            if existing:
                self._conn.execute(
                    """
                    UPDATE dev_product_events
                    SET last_seen_at = ?, client_ts = ?, app_version = ?, session_id = ?,
                        message_redacted = ?, context = ?, count = count + 1
                    WHERE signature = ?
                    """,
                    (
                        now,
                        event.get("client_ts"),
                        event.get("app_version"),
                        event.get("session_id"),
                        event.get("message_redacted"),
                        _json(event.get("context") or {}),
                        event["signature"],
                    ),
                )
                event_id = existing["event_id"]
            else:
                event_id = event["event_id"]
                self._conn.execute(
                    """
                    INSERT INTO dev_product_events (
                        event_id, received_at, last_seen_at, client_ts, type, app_version,
                        session_id, signature, message_redacted, context, count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        now,
                        now,
                        event.get("client_ts"),
                        event["type"],
                        event.get("app_version"),
                        event.get("session_id"),
                        event["signature"],
                        event.get("message_redacted"),
                        _json(event.get("context") or {}),
                        1,
                    ),
                )
        return self.get_event(event_id) or event

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_product_events WHERE event_id = ?",
            (str(event_id or "").strip(),),
        ).fetchone()
        return _event_from_row(row) if row else None

    def list_events(
        self,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[Dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if start is not None:
            clauses.append("last_seen_at >= ?")
            params.append(float(start))
        if end is not None:
            clauses.append("received_at <= ?")
            params.append(float(end))
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 100), 500)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_product_events
            {where}
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_event_from_row(row) for row in rows]


def normalize_product_event(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("event must be an object")
    event_type = str(raw.get("type") or "").strip()
    if event_type not in PRODUCT_EVENT_TYPES:
        raise ValueError(f"unsupported product event type: {event_type or '<empty>'}")
    event_id = str(raw.get("event_id") or raw.get("id") or "").strip()
    if not event_id:
        raise ValueError("event_id is required")
    context = _normalized_context(raw.get("context") or raw.get("details") or {})
    app_version = _bounded(raw.get("app_version"), 80)
    session_id = _bounded(raw.get("session_id"), 120)
    message_redacted = redact_product_message(raw.get("message_redacted") or raw.get("message") or "")
    return {
        "event_id": event_id[:160],
        "client_ts": _float_or_none(raw.get("client_ts") or raw.get("timestamp")),
        "type": event_type,
        "app_version": app_version,
        "session_id": session_id,
        "signature": product_event_signature(event_type, context),
        "message_redacted": message_redacted,
        "context": context,
    }


def product_event_signature(event_type: str, context: Dict[str, Any]) -> str:
    stable = {
        "type": str(event_type or "").strip(),
        "error_type": str(context.get("error_type") or context.get("exception_name") or "").strip(),
        "location": str(context.get("location") or context.get("route") or context.get("endpoint") or "").strip(),
        "flow": str(context.get("flow") or "").strip(),
        "screen": str(context.get("screen") or "").strip(),
        "status": str(context.get("status") or "").strip(),
    }
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def redact_product_message(value: Any) -> str:
    text = str(value or "").strip()
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_MESSAGE_CHARS]


def _normalized_context(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    context: Dict[str, str] = {}
    for key, raw in list(value.items())[:MAX_CONTEXT_KEYS]:
        key_text = str(key or "").strip().lower()
        if not key_text or _drops_context_key(key_text):
            continue
        context[key_text[:64]] = redact_product_message(raw)[:MAX_CONTEXT_VALUE_CHARS]
    return context


def _drops_context_key(key: str) -> bool:
    return any(part in key for part in ("prompt", "body", "content", "user_text", "input_text", "authorization", "api_key", "token", "secret"))


def _event_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "received_at": float(row["received_at"]),
        "last_seen_at": float(row["last_seen_at"]),
        "client_ts": float(row["client_ts"]) if row["client_ts"] is not None else None,
        "type": row["type"],
        "app_version": row["app_version"],
        "session_id": row["session_id"],
        "signature": row["signature"],
        "message_redacted": row["message_redacted"],
        "context": json.loads(row["context"] or "{}"),
        "count": int(row["count"] or 0),
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bounded(value: Any, limit: int) -> Optional[str]:
    text = str(value or "").strip()
    return text[:limit] if text else None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
