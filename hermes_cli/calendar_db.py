"""Judy calendar storage.

This module is intentionally small and synchronous, matching the existing
Gateway/tooling style. The calendar uses PostgreSQL when configured and falls
back to a profile-local SQLite database otherwise.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home


VALID_STATUSES = {"pending", "firing", "done", "cancelled"}
VALID_RECURRENCES = {"daily", "weekly", "monthly"}
STALE_FIRING_SECONDS = 30 * 60


def _dsn() -> str:
    return (
        os.environ.get("HERMES_CALENDAR_POSTGRES_DSN", "").strip()
        or os.environ.get("HERMES_KANBAN_POSTGRES_DSN", "").strip()
    )


def _connect():
    dsn = _dsn()
    if not dsn:
        conn = _connect_sqlite_without_schema()
        ensure_schema(conn)
        return conn
    try:
        import psycopg
        from psycopg.rows import dict_row
    except Exception as exc:
        raise RuntimeError(f"Calendar PostgreSQL runtime requires psycopg: {exc}") from exc
    conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
    ensure_schema(conn)
    return conn


def _using_postgres() -> bool:
    return bool(_dsn())


def _sqlite_path() -> Path:
    return get_hermes_home() / "data" / "judy_calendar.db"


def _connect_sqlite_without_schema() -> sqlite3.Connection:
    path = _sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _jsonb(value: dict[str, Any]):
    try:
        from psycopg.types.json import Jsonb
    except Exception:
        return json.dumps(value)
    return Jsonb(value)


def ensure_schema(conn: Any | None = None) -> None:
    own_conn = conn is None
    if conn is None:
        conn = _connect_without_schema() if _using_postgres() else _connect_sqlite_without_schema()
    ddl = _POSTGRES_SCHEMA if _using_postgres() else _SQLITE_SCHEMA
    try:
        if _using_postgres():
            with conn.cursor() as cur:
                cur.execute(ddl)
        else:
            conn.executescript(ddl)
    finally:
        if own_conn:
            conn.close()


_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS judy_calendar (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    scheduled_at TIMESTAMPTZ NOT NULL,
    recurrence TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_by TEXT DEFAULT 'judy',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fired_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    session_id TEXT,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_judy_calendar_status_due
    ON judy_calendar (status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_judy_calendar_tags
    ON judy_calendar USING GIN (tags);
"""


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS judy_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    scheduled_at TEXT NOT NULL,
    recurrence TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_by TEXT DEFAULT 'judy',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    fired_at TEXT,
    completed_at TEXT,
    session_id TEXT,
    context TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_judy_calendar_status_due
    ON judy_calendar (status, scheduled_at);
"""


def _connect_without_schema():
    dsn = _dsn()
    if not dsn:
        raise RuntimeError(
            "Calendar PostgreSQL runtime requires HERMES_CALENDAR_POSTGRES_DSN "
            "or HERMES_KANBAN_POSTGRES_DSN"
        )
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn, row_factory=dict_row, autocommit=True)


def _parse_dt(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return dt.astimezone(timezone.utc)


def _dt_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_recurrence(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in VALID_RECURRENCES:
        raise ValueError("recurrence must be one of: daily, weekly, monthly")
    return text


def _normalize_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in VALID_STATUSES:
        raise ValueError("invalid calendar status")
    return text


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [p.strip() for p in value.split(",")]
    elif isinstance(value, Iterable):
        items = [str(p).strip() for p in value]
    else:
        raise ValueError("tags must be a list of strings")
    return [p for p in items if p]


def _normalize_context(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("context must be a JSON object") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("context must be a JSON object")


def _row_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    for key in ("scheduled_at", "created_at", "fired_at", "completed_at"):
        val = d.get(key)
        if isinstance(val, datetime):
            d[key] = val.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        elif isinstance(val, str) and val:
            d[key] = _dt_text(_parse_dt(val, field=key))
    context = d.get("context") or {}
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            context = {}
    d["context"] = context if isinstance(context, dict) else {}
    tags = d.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    d["tags"] = [str(tag) for tag in tags] if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)) else []
    return d


def add_event(
    *,
    title: str,
    scheduled_at: Any,
    description: Optional[str] = None,
    recurrence: Optional[str] = None,
    tags: Any = None,
    context: Any = None,
    created_by: str = "judy",
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    title = str(title or "").strip()
    if not title:
        raise ValueError("title is required")
    scheduled = _parse_dt(scheduled_at, field="scheduled_at")
    recurrence_norm = _normalize_recurrence(recurrence)
    tags_norm = _normalize_tags(tags)
    context_norm = _normalize_context(context)
    if not _using_postgres():
        with _connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO judy_calendar
                    (title, description, scheduled_at, recurrence, tags, context,
                     created_by, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING *
                """,
                (
                    title,
                    description,
                    _dt_text(scheduled),
                    recurrence_norm,
                    json.dumps(tags_norm),
                    json.dumps(context_norm),
                    created_by or "judy",
                    session_id,
                ),
            )
            return _row_dict(cur.fetchone())
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO judy_calendar
                    (title, description, scheduled_at, recurrence, tags, context,
                     created_by, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    title,
                    description,
                    scheduled,
                    recurrence_norm,
                    tags_norm,
                    _jsonb(context_norm),
                    created_by or "judy",
                    session_id,
                ),
            )
            return _row_dict(cur.fetchone())


def list_events(
    *,
    status: Optional[str] = None,
    from_: Any = None,
    to: Any = None,
    tags: Any = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    status_norm = _normalize_status(status)
    from_dt = _parse_dt(from_, field="from") if from_ else None
    to_dt = _parse_dt(to, field="to") if to else None
    tags_norm = _normalize_tags(tags) if tags is not None else []
    limit = max(1, min(int(limit or 50), 200))
    clauses: list[str] = []
    params: list[Any] = []
    placeholder = "%s" if _using_postgres() else "?"
    if status_norm:
        clauses.append(f"status = {placeholder}")
        params.append(status_norm)
    if from_dt:
        clauses.append(f"scheduled_at >= {placeholder}")
        params.append(from_dt)
    if to_dt:
        clauses.append(f"scheduled_at <= {placeholder}")
        params.append(to_dt)
    if tags_norm:
        if _using_postgres():
            clauses.append("tags && %s")
            params.append(tags_norm)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    if not _using_postgres():
        sqlite_params = [_dt_text(p) if isinstance(p, datetime) else p for p in params]
        query_limit = "" if tags_norm else "LIMIT ?"
        if not tags_norm:
            sqlite_params.append(limit)
        with _connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM judy_calendar
                {where}
                ORDER BY scheduled_at ASC, id ASC
                {query_limit}
                """,
                tuple(sqlite_params),
            ).fetchall()
        normalized = [_row_dict(row) for row in rows]
        if tags_norm:
            wanted = set(tags_norm)
            normalized = [row for row in normalized if wanted.intersection(row.get("tags") or [])]
        return normalized[:limit]
    params.append(limit)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM judy_calendar
                {where}
                ORDER BY scheduled_at ASC, id ASC
                LIMIT %s
                """,
                tuple(params),
            )
            return [_row_dict(row) for row in cur.fetchall()]


def upcoming_events(*, limit: int = 5) -> list[dict[str, Any]]:
    return list_events(status="pending", from_=datetime.now(timezone.utc), limit=limit)


def get_event(event_id: int) -> Optional[dict[str, Any]]:
    if not _using_postgres():
        with _connect() as conn:
            row = conn.execute("SELECT * FROM judy_calendar WHERE id = ?", (int(event_id),)).fetchone()
            return _row_dict(row) if row else None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM judy_calendar WHERE id = %s", (int(event_id),))
            row = cur.fetchone()
            return _row_dict(row) if row else None


def update_event(event_id: int, **updates: Any) -> Optional[dict[str, Any]]:
    allowed = {"title", "scheduled_at", "description", "tags", "context", "recurrence"}
    fields: list[str] = []
    params: list[Any] = []
    for key, value in updates.items():
        if key not in allowed or value is None:
            continue
        if key == "title":
            value = str(value).strip()
            if not value:
                raise ValueError("title cannot be empty")
        elif key == "scheduled_at":
            value = _parse_dt(value, field="scheduled_at")
        elif key == "tags":
            value = _normalize_tags(value)
        elif key == "context":
            value = _normalize_context(value)
        elif key == "recurrence":
            value = _normalize_recurrence(value)
        if _using_postgres():
            fields.append(f"{key} = %s")
            params.append(_jsonb(value) if key == "context" else value)
        else:
            fields.append(f"{key} = ?")
            if key == "scheduled_at":
                params.append(_dt_text(value))
            elif key in {"tags", "context"}:
                params.append(json.dumps(value))
            else:
                params.append(value)
    if not fields:
        return get_event(event_id)
    params.append(int(event_id))
    if not _using_postgres():
        with _connect() as conn:
            cur = conn.execute(
                f"UPDATE judy_calendar SET {', '.join(fields)} WHERE id = ? RETURNING *",
                tuple(params),
            )
            row = cur.fetchone()
            return _row_dict(row) if row else None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE judy_calendar SET {', '.join(fields)} WHERE id = %s RETURNING *",
                tuple(params),
            )
            row = cur.fetchone()
            return _row_dict(row) if row else None


def cancel_event(event_id: int) -> Optional[dict[str, Any]]:
    if not _using_postgres():
        with _connect() as conn:
            row = conn.execute(
                """
                UPDATE judy_calendar
                SET status = 'cancelled'
                WHERE id = ? AND status != 'cancelled'
                RETURNING *
                """,
                (int(event_id),),
            ).fetchone()
            return _row_dict(row) if row else None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE judy_calendar
                SET status = 'cancelled'
                WHERE id = %s AND status != 'cancelled'
                RETURNING *
                """,
                (int(event_id),),
            )
            row = cur.fetchone()
            return _row_dict(row) if row else None


def _advance(dt: datetime, recurrence: str) -> datetime:
    if recurrence == "daily":
        return dt + timedelta(days=1)
    if recurrence == "weekly":
        return dt + timedelta(weeks=1)
    if recurrence == "monthly":
        month = dt.month + 1
        year = dt.year
        if month > 12:
            month = 1
            year += 1
        day = min(dt.day, _days_in_month(year, month))
        return dt.replace(year=year, month=month, day=day)
    return dt


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    cur = datetime(year, month, 1, tzinfo=timezone.utc)
    return (nxt - cur).days


def mark_done(event_id: int, notes: Optional[str] = None, session_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    event = get_event(event_id)
    if not event:
        return None
    recurrence = event.get("recurrence")
    completed = datetime.now(timezone.utc)
    scheduled = _parse_dt(event["scheduled_at"], field="scheduled_at")
    if not _using_postgres():
        with _connect() as conn:
            if recurrence:
                next_at = _advance(scheduled, recurrence)
                while next_at <= completed:
                    next_at = _advance(next_at, recurrence)
                row = conn.execute(
                    """
                    UPDATE judy_calendar
                    SET status = 'pending',
                        scheduled_at = ?,
                        fired_at = NULL,
                        completed_at = ?,
                        notes = ?,
                        session_id = COALESCE(?, session_id)
                    WHERE id = ?
                    RETURNING *
                    """,
                    (_dt_text(next_at), _dt_text(completed), notes, session_id, int(event_id)),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    UPDATE judy_calendar
                    SET status = 'done',
                        completed_at = ?,
                        notes = ?,
                        session_id = COALESCE(?, session_id)
                    WHERE id = ?
                    RETURNING *
                    """,
                    (_dt_text(completed), notes, session_id, int(event_id)),
                ).fetchone()
            return _row_dict(row) if row else None
    with _connect() as conn:
        with conn.cursor() as cur:
            if recurrence:
                next_at = _advance(scheduled, recurrence)
                while next_at <= completed:
                    next_at = _advance(next_at, recurrence)
                cur.execute(
                    """
                    UPDATE judy_calendar
                    SET status = 'pending',
                        scheduled_at = %s,
                        fired_at = NULL,
                        completed_at = %s,
                        notes = %s,
                        session_id = COALESCE(%s, session_id)
                    WHERE id = %s
                    RETURNING *
                    """,
                    (next_at, completed, notes, session_id, int(event_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE judy_calendar
                    SET status = 'done',
                        completed_at = %s,
                        notes = %s,
                        session_id = COALESCE(%s, session_id)
                    WHERE id = %s
                    RETURNING *
                    """,
                    (completed, notes, session_id, int(event_id)),
                )
            row = cur.fetchone()
            return _row_dict(row) if row else None


def claim_due_events(*, now: Any = None, limit: int = 10) -> list[dict[str, Any]]:
    now_dt = _parse_dt(now, field="now") if now else datetime.now(timezone.utc)
    limit = max(1, min(int(limit or 10), 50))
    if not _using_postgres():
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            due_rows = conn.execute(
                """
                SELECT id FROM judy_calendar
                WHERE status = 'pending' AND scheduled_at <= ?
                ORDER BY scheduled_at ASC, id ASC
                LIMIT ?
                """,
                (_dt_text(now_dt), limit),
            ).fetchall()
            ids = [int(row["id"]) for row in due_rows]
            if not ids:
                conn.commit()
                return []
            placeholders = ", ".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE judy_calendar
                SET status = 'firing', fired_at = ?
                WHERE id IN ({placeholders})
                """,
                (_dt_text(now_dt), *ids),
            )
            rows = conn.execute(
                f"""
                SELECT * FROM judy_calendar
                WHERE id IN ({placeholders})
                ORDER BY scheduled_at ASC, id ASC
                """,
                tuple(ids),
            ).fetchall()
            conn.commit()
            return [_row_dict(row) for row in rows]
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH due AS (
                    SELECT id FROM judy_calendar
                    WHERE status = 'pending' AND scheduled_at <= %s
                    ORDER BY scheduled_at ASC, id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE judy_calendar c
                SET status = 'firing', fired_at = %s
                FROM due
                WHERE c.id = due.id
                RETURNING c.*
                """,
                (now_dt, limit, now_dt),
            )
            return [_row_dict(row) for row in cur.fetchall()]


def requeue_stale_firing(*, older_than_seconds: int = STALE_FIRING_SECONDS) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, int(older_than_seconds)))
    if not _using_postgres():
        with _connect() as conn:
            cur = conn.execute(
                """
                UPDATE judy_calendar
                SET status = 'pending', fired_at = NULL
                WHERE status = 'firing' AND fired_at < ?
                """,
                (_dt_text(cutoff),),
            )
            return int(cur.rowcount or 0)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE judy_calendar
                SET status = 'pending', fired_at = NULL
                WHERE status = 'firing' AND fired_at < %s
                """,
                (cutoff,),
            )
            return int(cur.rowcount or 0)


def release_claim(event_id: int) -> Optional[dict[str, Any]]:
    """Return a firing event to pending when Gateway could not dispatch it."""
    if not _using_postgres():
        with _connect() as conn:
            row = conn.execute(
                """
                UPDATE judy_calendar
                SET status = 'pending', fired_at = NULL
                WHERE id = ? AND status = 'firing'
                RETURNING *
                """,
                (int(event_id),),
            ).fetchone()
            return _row_dict(row) if row else None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE judy_calendar
                SET status = 'pending', fired_at = NULL
                WHERE id = %s AND status = 'firing'
                RETURNING *
                """,
                (int(event_id),),
            )
            row = cur.fetchone()
            return _row_dict(row) if row else None
