"""Persistent outbound message delivery log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

VALID_SEND_TYPES = frozenset(
    {
        "feishu_real_send",
        "local_relay",
        "draft_only",
        "blocked_by_validator",
    }
)

_PREVIEW_LIMIT = 500
_SUMMARY_LIMIT = 1000

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS outbound_log (
    outbound_id          TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    platform             TEXT NOT NULL,
    chat_id              TEXT NOT NULL,
    chat_type            TEXT,
    source_message_id    TEXT,
    reply_to_message_id  TEXT,
    send_type            TEXT NOT NULL,
    workflow_id          TEXT,
    task_id              TEXT,
    to_role              TEXT,
    target_role          TEXT,
    target_profile       TEXT,
    content_hash         TEXT NOT NULL,
    content_preview      TEXT NOT NULL,
    send_success         INTEGER NOT NULL CHECK (send_success IN (0, 1)),
    feishu_message_id    TEXT,
    real_sent            INTEGER NOT NULL DEFAULT 0 CHECK (real_sent IN (0, 1)),
    error                TEXT,
    raw_response_summary TEXT,
    CHECK (send_type IN ('feishu_real_send', 'local_relay', 'draft_only', 'blocked_by_validator'))
);

CREATE INDEX IF NOT EXISTS idx_outbound_log_chat_created
    ON outbound_log (chat_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_outbound_log_success_created
    ON outbound_log (send_success, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_outbound_log_task_created
    ON outbound_log (workflow_id, task_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_outbound_log_type_created
    ON outbound_log (send_type, created_at DESC);
"""


def _db_path() -> Path:
    return get_hermes_home() / "outbound_log.db"


def _get_conn() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    _migrate_schema(conn)
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(outbound_log)").fetchall()}
    additions = {
        "workflow_id": "ALTER TABLE outbound_log ADD COLUMN workflow_id TEXT",
        "task_id": "ALTER TABLE outbound_log ADD COLUMN task_id TEXT",
        "to_role": "ALTER TABLE outbound_log ADD COLUMN to_role TEXT",
        "real_sent": "ALTER TABLE outbound_log ADD COLUMN real_sent INTEGER NOT NULL DEFAULT 0 CHECK (real_sent IN (0, 1))",
    }
    for column, sql in additions.items():
        if column not in columns:
            conn.execute(sql)
    conn.commit()


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _preview(content: Any, *, limit: int = _PREVIEW_LIMIT) -> str:
    text = "" if content is None else str(content)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _content_hash(content: Any) -> str:
    text = "" if content is None else str(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    data["send_success"] = bool(data["send_success"])
    data["real_sent"] = bool(data.get("real_sent", False))
    return data


def summarize_raw_response(raw_response: Any, *, limit: int = _SUMMARY_LIMIT) -> Optional[str]:
    """Return a compact, best-effort summary of a platform API response."""
    if raw_response is None:
        return None

    summary: Dict[str, Any] = {}
    success = getattr(raw_response, "success", None)
    if callable(success):
        try:
            summary["success"] = bool(success())
        except Exception:
            pass
    for attr in ("code", "msg", "request_id"):
        value = getattr(raw_response, attr, None)
        if value is not None:
            summary[attr] = value
    data = getattr(raw_response, "data", None)
    if data is not None:
        data_summary: Dict[str, Any] = {}
        for attr in ("message_id", "open_message_id", "chat_id"):
            value = getattr(data, attr, None)
            if value is not None:
                data_summary[attr] = value
        if data_summary:
            summary["data"] = data_summary

    if summary:
        text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    else:
        text = repr(raw_response)

    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def record_outbound(
    *,
    platform: str,
    chat_id: str,
    send_type: str,
    send_success: bool,
    content: Any = None,
    outbound_id: Optional[str] = None,
    created_at: Optional[str] = None,
    chat_type: Optional[str] = None,
    source_message_id: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    task_id: Optional[str] = None,
    to_role: Optional[str] = None,
    target_role: Optional[str] = None,
    target_profile: Optional[str] = None,
    content_hash: Optional[str] = None,
    content_preview: Optional[str] = None,
    feishu_message_id: Optional[str] = None,
    real_sent: Optional[bool] = None,
    error: Optional[str] = None,
    raw_response_summary: Optional[str] = None,
) -> str:
    """Insert one outbound delivery attempt and return its outbound_id."""
    if send_type not in VALID_SEND_TYPES:
        raise ValueError(f"Unsupported send_type: {send_type}")

    outbound_id = outbound_id or f"out_{uuid.uuid4().hex}"
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    content_for_hash = content if content is not None else content_preview
    content_hash = content_hash or _content_hash(content_for_hash)
    content_preview = content_preview if content_preview is not None else _preview(content)
    resolved_real_sent = (
        bool(real_sent)
        if real_sent is not None
        else bool(send_success and send_type == "feishu_real_send" and feishu_message_id)
    )

    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO outbound_log (
                outbound_id,
                created_at,
                platform,
                chat_id,
                chat_type,
                source_message_id,
                reply_to_message_id,
                send_type,
                workflow_id,
                task_id,
                to_role,
                target_role,
                target_profile,
                content_hash,
                content_preview,
                send_success,
                feishu_message_id,
                real_sent,
                error,
                raw_response_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outbound_id,
                created_at,
                str(platform),
                str(chat_id),
                _coerce_text(chat_type),
                _coerce_text(source_message_id),
                _coerce_text(reply_to_message_id),
                send_type,
                _coerce_text(workflow_id),
                _coerce_text(task_id),
                _coerce_text(to_role),
                _coerce_text(target_role),
                _coerce_text(target_profile),
                content_hash,
                content_preview,
                1 if send_success else 0,
                _coerce_text(feishu_message_id),
                1 if resolved_real_sent else 0,
                _coerce_text(error),
                _coerce_text(raw_response_summary),
            ),
        )
        conn.commit()
        return outbound_id
    finally:
        conn.close()


def get_outbound(outbound_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single outbound record by id."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM outbound_log WHERE outbound_id = ?",
            (outbound_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_outbounds(
    chat_id: Optional[str] = None,
    send_success: Optional[bool] = None,
    workflow_id: Optional[str] = None,
    task_id: Optional[str] = None,
    send_type: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List recent outbound records, optionally filtered by chat or success."""
    clauses: List[str] = []
    params: List[Any] = []
    if chat_id is not None:
        clauses.append("chat_id = ?")
        params.append(str(chat_id))
    if send_success is not None:
        clauses.append("send_success = ?")
        params.append(1 if send_success else 0)
    if workflow_id is not None:
        clauses.append("workflow_id = ?")
        params.append(str(workflow_id))
    if task_id is not None:
        clauses.append("task_id = ?")
        params.append(str(task_id))
    if send_type is not None:
        clauses.append("send_type = ?")
        params.append(str(send_type))

    try:
        parsed_limit = int(limit)
    except (TypeError, ValueError):
        parsed_limit = 50
    parsed_limit = min(max(parsed_limit, 1), 1000)

    sql = "SELECT * FROM outbound_log"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(parsed_limit)

    conn = _get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()
