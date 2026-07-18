"""Local Hermes Agent Timeline observability plugin.

Stores read-only observer-hook events in ``~/.hermes/timelines/timeline.db`` so
operators can inspect what happened inside an agent run without sending data to
a third-party observability service.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_DB_LOCK = threading.RLock()
_PENDING_TOOL_STARTS: dict[str, float] = {}
_PENDING_API_STARTS: dict[str, float] = {}
_PENDING_TURN_STARTS: dict[str, float] = {}

_SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "auth", "bearer", "client_secret",
    "password", "private_key", "refresh_token", "secret", "token",
}
_MAX_TEXT_CHARS = 1200
_MAX_PAYLOAD_CHARS = 8000


def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or _now(), tz=timezone.utc).isoformat()


def _db_path() -> Path:
    base = get_hermes_home() / "timelines"
    base.mkdir(parents=True, exist_ok=True)
    return base / "timeline.db"


def _db_file_bytes(path: Optional[Path] = None) -> int:
    db = path or _db_path()
    total = 0
    for candidate in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
        try:
            total += candidate.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _connect(path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or _db_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');

        CREATE TABLE IF NOT EXISTS timeline_runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT,
            turn_id TEXT,
            task_id TEXT,
            platform TEXT,
            source TEXT,
            model TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            started_at REAL NOT NULL,
            ended_at REAL,
            duration_ms INTEGER,
            summary TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS timeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts REAL NOT NULL,
            event_type TEXT NOT NULL,
            name TEXT,
            status TEXT,
            duration_ms INTEGER,
            summary TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(run_id) REFERENCES timeline_runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_timeline_runs_started_at
            ON timeline_runs(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_timeline_runs_session_turn
            ON timeline_runs(session_id, turn_id);
        CREATE INDEX IF NOT EXISTS idx_timeline_events_run_ts
            ON timeline_events(run_id, ts, id);
        CREATE INDEX IF NOT EXISTS idx_timeline_events_type
            ON timeline_events(event_type);
        """
    )
    conn.commit()


def _ensure_db() -> sqlite3.Connection:
    conn = _connect()
    _init_db(conn)
    return conn


def _run_id(session_id: str = "", turn_id: str = "", task_id: str = "", api_request_id: str = "") -> str:
    if turn_id:
        return f"turn:{turn_id}"
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    if api_request_id:
        return f"api:{api_request_id}"
    return f"process:{threading.get_ident()}"


def _short_text(value: Any, limit: int = _MAX_TEXT_CHARS) -> Any:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(_safe_value(value), ensure_ascii=False, default=str)
    try:
        from agent.redact import redact_sensitive_text
        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    if len(text) > limit:
        return text[:limit] + f"… [truncated {len(text) - limit} chars]"
    return text


def _safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _short_text(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:80]:
            key = str(k)
            if key.lower() in _SENSITIVE_KEYS or any(s in key.lower() for s in ("token", "secret", "password", "api_key")):
                out[key] = "[REDACTED]"
            else:
                out[key] = _safe_value(v, depth + 1)
        if len(value) > 80:
            out["<truncated_keys>"] = len(value) - 80
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out_list: list[Any] = [_safe_value(v, depth + 1) for v in seq[:80]]
        if len(seq) > 80:
            out_list.append({"<truncated_items>": len(seq) - 80})
        return out_list
    if hasattr(value, "__dict__"):
        try:
            return _safe_value(vars(value), depth + 1)
        except Exception:
            pass
    return _short_text(repr(value))


def _payload_json(payload: dict[str, Any]) -> str:
    safe = _safe_value(payload)
    text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) > _MAX_PAYLOAD_CHARS:
        safe = {"truncated": True, "chars": len(text), "preview": text[:_MAX_PAYLOAD_CHARS]}
        text = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    return text


def _upsert_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    session_id: str = "",
    turn_id: str = "",
    task_id: str = "",
    platform: str = "",
    source: str = "",
    model: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    ts = _now()
    conn.execute(
        """
        INSERT INTO timeline_runs(run_id, session_id, turn_id, task_id, platform, source, model, started_at, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            session_id=COALESCE(NULLIF(excluded.session_id, ''), timeline_runs.session_id),
            turn_id=COALESCE(NULLIF(excluded.turn_id, ''), timeline_runs.turn_id),
            task_id=COALESCE(NULLIF(excluded.task_id, ''), timeline_runs.task_id),
            platform=COALESCE(NULLIF(excluded.platform, ''), timeline_runs.platform),
            source=COALESCE(NULLIF(excluded.source, ''), timeline_runs.source),
            model=COALESCE(NULLIF(excluded.model, ''), timeline_runs.model)
        """,
        (run_id, session_id, turn_id, task_id, platform, source, model, ts, _payload_json(metadata or {})),
    )


def _add_event(
    *,
    event_type: str,
    name: str = "",
    status: str = "",
    duration_ms: Any = None,
    summary: str = "",
    payload: Optional[dict[str, Any]] = None,
    session_id: str = "",
    turn_id: str = "",
    task_id: str = "",
    api_request_id: str = "",
    platform: str = "",
    source: str = "",
    model: str = "",
) -> str:
    rid = _run_id(session_id=session_id, turn_id=turn_id, task_id=task_id, api_request_id=api_request_id)
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            _upsert_run(
                conn,
                run_id=rid,
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                platform=platform,
                source=source,
                model=model,
                metadata={"schema_version": _SCHEMA_VERSION},
            )
            conn.execute(
                """
                INSERT INTO timeline_events(run_id, ts, event_type, name, status, duration_ms, summary, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, _now(), event_type, name, status, _coerce_duration_ms(duration_ms), _short_text(summary, 500) or "", _payload_json(payload or {})),
            )
            conn.commit()
        finally:
            conn.close()
    return rid


def _finish_run(run_id: str, *, status: str = "completed", summary: str = "") -> None:
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            row = conn.execute("SELECT started_at FROM timeline_runs WHERE run_id = ?", (run_id,)).fetchone()
            ended = _now()
            duration_ms = int((ended - float(row["started_at"])) * 1000) if row else None
            conn.execute(
                "UPDATE timeline_runs SET status = ?, ended_at = ?, duration_ms = ?, summary = ? WHERE run_id = ?",
                (status, ended, duration_ms, _short_text(summary, 500) or "", run_id),
            )
            conn.commit()
        finally:
            conn.close()


def _duration_since(key: str, store: dict[str, float]) -> Optional[int]:
    started = store.pop(key, None)
    if started is None:
        return None
    return int((_now() - started) * 1000)


def _coerce_duration_ms(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _platform_from_event(event: Any) -> tuple[str, str, dict[str, Any]]:
    source = getattr(event, "source", None)
    platform_obj = getattr(source, "platform", None)
    platform = getattr(platform_obj, "value", None) or str(platform_obj or "")
    payload = {
        "chat_id": getattr(source, "chat_id", None),
        "thread_id": getattr(source, "thread_id", None),
        "message_id": getattr(source, "message_id", None),
        "chat_type": getattr(source, "chat_type", None),
        "user_id": getattr(source, "user_id", None),
        "user_name": getattr(source, "user_name", None),
        "text_preview": _short_text(getattr(event, "text", ""), 500),
    }
    source_key = ":".join(str(x) for x in (platform, payload.get("chat_id") or "", payload.get("thread_id") or "") if x)
    return platform, source_key, payload


# ---- hook callbacks -------------------------------------------------------

def on_pre_gateway_dispatch(*, event: Any = None, **_: Any) -> None:
    platform, source, payload = _platform_from_event(event)
    _add_event(event_type="gateway.message", name="pre_gateway_dispatch", status="received", payload=payload, platform=platform, source=source)


def on_post_gateway_delivery(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    status = kwargs.get("status") or ("ok" if kwargs.get("success") else "error")
    operation = str(kwargs.get("operation") or "send")
    message_id = str(kwargs.get("message_id") or "")
    summary_bits = [operation, status]
    if message_id:
        summary_bits.append(f"message={message_id}")
    error_text = kwargs.get("error")
    _add_event(
        event_type="gateway.delivery",
        name=f"delivery_{operation}",
        status=status,
        summary=_short_text(error_text, 500) if status == "error" and error_text else " ".join(summary_bits),
        payload={
            k: kwargs.get(k)
            for k in (
                "operation", "success", "message_id", "error", "content_preview",
                "content_chars", "metadata", "finalize", "transport",
                "raw_response", "continuation_message_ids", "chat_type",
            )
            if k in kwargs
        },
        **ids,
    )


def on_session_start(**kwargs: Any) -> None:
    _add_event(event_type="session", name="session_start", status="started", summary="Session started", payload=kwargs, **_ids(kwargs))


def on_session_end(**kwargs: Any) -> None:
    status = "completed" if kwargs.get("completed", True) else "interrupted"
    rid = _add_event(event_type="session", name="session_end", status=status, summary=kwargs.get("reason") or "Session ended", payload=kwargs, **_ids(kwargs))
    _finish_run(rid, status=status, summary=kwargs.get("reason") or "")
    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        _finish_run(f"session:{session_id}", status=status, summary=kwargs.get("reason") or "Session ended")


def on_session_finalize(**kwargs: Any) -> None:
    _add_event(event_type="session", name="session_finalize", status="finalized", summary=kwargs.get("reason") or "Session finalized", payload=kwargs, **_ids(kwargs))


def on_session_reset(**kwargs: Any) -> None:
    _add_event(event_type="session", name="session_reset", status="reset", summary=kwargs.get("reason") or "Session reset", payload=kwargs, **_ids(kwargs))


def on_pre_llm_call(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    key = ids.get("turn_id") or ids.get("session_id") or ids.get("task_id") or ""
    if key:
        _PENDING_TURN_STARTS[key] = _now()
    _add_event(
        event_type="turn",
        name="user_message",
        status="started",
        summary=_short_text(kwargs.get("user_message") or kwargs.get("messages") or kwargs.get("conversation_history"), 500) or "Agent turn started",
        payload={k: kwargs.get(k) for k in ("user_message", "is_first_turn", "turn_type", "message_count") if k in kwargs},
        **ids,
    )


def on_post_llm_turn(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    key = ids.get("turn_id") or ids.get("session_id") or ids.get("task_id") or ""
    duration_ms = _duration_since(key, _PENDING_TURN_STARTS) if key else None
    rid = _add_event(
        event_type="turn",
        name="assistant_response",
        status="completed",
        duration_ms=duration_ms,
        summary=_short_text(kwargs.get("assistant_response") or kwargs.get("assistant_message") or kwargs.get("response"), 500) or "Assistant response",
        payload={k: kwargs.get(k) for k in ("assistant_response", "model", "platform") if k in kwargs},
        **ids,
    )
    _finish_run(rid, status="completed", summary="Turn completed")


def on_pre_api_request(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    api_id = ids.get("api_request_id") or f"{ids.get('turn_id')}:{kwargs.get('api_call_count', '')}"
    if api_id:
        _PENDING_API_STARTS[api_id] = _now()
    _add_event(
        event_type="llm.request",
        name="api_request",
        status="started",
        summary=f"{kwargs.get('provider') or ''} {kwargs.get('model') or ''}".strip(),
        payload={k: kwargs.get(k) for k in ("provider", "base_url", "api_mode", "api_call_count", "message_count", "tool_count", "approx_input_tokens", "max_tokens") if k in kwargs},
        **ids,
    )


def on_post_api_request(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    api_id = ids.get("api_request_id") or f"{ids.get('turn_id')}:{kwargs.get('api_call_count', '')}"
    duration_ms = _duration_since(api_id, _PENDING_API_STARTS) if api_id else None
    if duration_ms is None and kwargs.get("api_duration") is not None:
        try:
            duration_ms = int(float(kwargs["api_duration"]) * 1000)
        except Exception:
            duration_ms = None
    _add_event(
        event_type="llm.response",
        name="api_response",
        status="completed",
        duration_ms=duration_ms,
        summary=f"finish={kwargs.get('finish_reason') or ''} tools={kwargs.get('assistant_tool_call_count') or 0}",
        payload={k: kwargs.get(k) for k in ("finish_reason", "usage", "assistant_content_chars", "assistant_tool_call_count", "response_model") if k in kwargs},
        **ids,
    )


def on_api_request_error(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    api_id = ids.get("api_request_id") or f"{ids.get('turn_id')}:{kwargs.get('api_call_count', '')}"
    duration_ms = _duration_since(api_id, _PENDING_API_STARTS) if api_id else None
    _add_event(
        event_type="llm.error",
        name="api_request_error",
        status="error",
        duration_ms=duration_ms,
        summary=_short_text(kwargs.get("reason") or kwargs.get("error"), 500) or "API request failed",
        payload=kwargs,
        **ids,
    )


def on_pre_tool_call(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    call_id = kwargs.get("tool_call_id") or f"{ids.get('turn_id')}:{kwargs.get('tool_name')}:{uuid.uuid4().hex[:8]}"
    _PENDING_TOOL_STARTS[str(call_id)] = _now()
    _add_event(
        event_type="tool.call",
        name=kwargs.get("tool_name") or "tool",
        status="started",
        summary=f"Tool call: {kwargs.get('tool_name') or 'tool'}",
        payload={"tool_call_id": call_id, "args": kwargs.get("args")},
        **ids,
    )


def on_post_tool_call(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    call_id = kwargs.get("tool_call_id") or ""
    duration_ms = _duration_since(str(call_id), _PENDING_TOOL_STARTS) if call_id else kwargs.get("duration_ms")
    _add_event(
        event_type="tool.result",
        name=kwargs.get("tool_name") or "tool",
        status=kwargs.get("status") or ("error" if kwargs.get("error_message") else "ok"),
        duration_ms=duration_ms,
        summary=_short_text(kwargs.get("error_message") or kwargs.get("result"), 500) or f"Tool result: {kwargs.get('tool_name') or 'tool'}",
        payload={"tool_call_id": call_id, "result": kwargs.get("result"), "error_type": kwargs.get("error_type"), "error_message": kwargs.get("error_message")},
        **ids,
    )


def on_pre_approval_request(**kwargs: Any) -> None:
    _add_event(event_type="approval", name="approval_request", status="pending", summary=kwargs.get("description") or kwargs.get("command") or "Approval requested", payload=kwargs, **_ids(kwargs))


def on_post_approval_response(**kwargs: Any) -> None:
    _add_event(event_type="approval", name="approval_response", status=kwargs.get("choice") or "answered", summary=kwargs.get("choice") or "Approval response", payload=kwargs, **_ids(kwargs))


def on_subagent_start(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    ids.setdefault("session_id", kwargs.get("parent_session_id") or "")
    _add_event(event_type="subagent", name="subagent_start", status="started", summary=_short_text(kwargs.get("child_goal"), 500) or "Subagent started", payload=kwargs, **ids)


def on_subagent_stop(**kwargs: Any) -> None:
    ids = _ids(kwargs)
    ids.setdefault("session_id", kwargs.get("parent_session_id") or "")
    _add_event(event_type="subagent", name="subagent_stop", status=kwargs.get("status") or "completed", duration_ms=kwargs.get("duration_ms"), summary=_short_text(kwargs.get("child_summary"), 500) or "Subagent stopped", payload=kwargs, **ids)


def _ids(kwargs: dict[str, Any]) -> dict[str, str]:
    platform = str(kwargs.get("platform") or "")
    chat_id = str(kwargs.get("chat_id") or "")
    thread_id = str(kwargs.get("thread_id") or "")
    source = str(kwargs.get("source") or kwargs.get("gateway_session_key") or "")
    if not source and platform and chat_id:
        parts = [platform, chat_id]
        if thread_id:
            parts.append(thread_id)
        source = ":".join(parts)
    return {
        "session_id": str(kwargs.get("session_id") or kwargs.get("parent_session_id") or ""),
        "turn_id": str(kwargs.get("turn_id") or kwargs.get("parent_turn_id") or ""),
        "task_id": str(kwargs.get("task_id") or kwargs.get("child_subagent_id") or ""),
        "api_request_id": str(kwargs.get("api_request_id") or ""),
        "platform": platform,
        "source": source,
        "model": str(kwargs.get("model") or kwargs.get("response_model") or ""),
    }


# ---- query + CLI ----------------------------------------------------------

def list_runs(limit: int = 20, status: str = "", session_id: str = "", platform: str = "", source: str = "") -> list[dict[str, Any]]:
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            where = []
            args: list[Any] = []
            if status:
                where.append("status = ?")
                args.append(status)
            if session_id:
                where.append("session_id = ?")
                args.append(session_id)
            if platform:
                where.append("platform = ?")
                args.append(platform)
            if source:
                where.append("source LIKE ?")
                args.append(source if "%" in source else source + "%")
            sql = "SELECT * FROM timeline_runs"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY started_at DESC LIMIT ?"
            args.append(limit)
            return [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()


def list_thread_runs(platform: str, chat_id: str, thread_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """List runs whose source includes a platform chat/thread route."""
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            where = []
            args: list[Any] = []
            if platform:
                where.append("platform = ?")
                args.append(platform)
            if chat_id:
                where.append("source LIKE ?")
                args.append(f"%{chat_id}%")
            if thread_id:
                where.append("source LIKE ?")
                args.append(f"%{thread_id}%")
            sql = "SELECT * FROM timeline_runs"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY started_at DESC LIMIT ?"
            args.append(limit)
            return [dict(r) for r in conn.execute(sql, args).fetchall()]
        finally:
            conn.close()


def get_run(run_id: str) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            run = conn.execute("SELECT * FROM timeline_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not run:
                # Prefix convenience: allow `hermes timeline show turn:abc...` shortened.
                matches = conn.execute("SELECT * FROM timeline_runs WHERE run_id LIKE ? ORDER BY started_at DESC LIMIT 2", (run_id + "%",)).fetchall()
                if len(matches) == 1:
                    run = matches[0]
            if not run:
                return None, []
            events = conn.execute("SELECT * FROM timeline_events WHERE run_id = ? ORDER BY ts, id", (run["run_id"],)).fetchall()
            return dict(run), [dict(e) for e in events]
        finally:
            conn.close()


def timeline_stats() -> dict[str, Any]:
    """Return storage and row-count stats for the local timeline database."""
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            runs = int(conn.execute("SELECT COUNT(*) FROM timeline_runs").fetchone()[0])
            events = int(conn.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0])
            oldest = conn.execute("SELECT MIN(started_at) FROM timeline_runs").fetchone()[0]
            newest = conn.execute("SELECT MAX(started_at) FROM timeline_runs").fetchone()[0]
            statuses = {
                str(row["status"] or ""): int(row["count"])
                for row in conn.execute("SELECT status, COUNT(*) AS count FROM timeline_runs GROUP BY status").fetchall()
            }
            platforms = {
                str(row["platform"] or ""): int(row["count"])
                for row in conn.execute("SELECT platform, COUNT(*) AS count FROM timeline_runs GROUP BY platform").fetchall()
            }
            return {
                "db_path": str(_db_path()),
                "db_bytes": _db_file_bytes(),
                "runs": runs,
                "events": events,
                "oldest_started_at": float(oldest) if oldest is not None else None,
                "oldest_started_at_iso": _iso(float(oldest)) if oldest is not None else "",
                "newest_started_at": float(newest) if newest is not None else None,
                "newest_started_at_iso": _iso(float(newest)) if newest is not None else "",
                "statuses": statuses,
                "platforms": platforms,
            }
        finally:
            conn.close()


def prune_timeline(*, days: int, dry_run: bool = False, vacuum: bool = False) -> dict[str, Any]:
    """Delete runs older than ``days`` days. Events cascade via FK."""
    if days < 1:
        raise ValueError("days must be >= 1")
    cutoff = _now() - (days * 86400)
    before_bytes = _db_file_bytes()
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            run_ids = [
                str(row["run_id"])
                for row in conn.execute("SELECT run_id FROM timeline_runs WHERE started_at < ?", (cutoff,)).fetchall()
            ]
            event_count = 0
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                event_count = int(conn.execute(f"SELECT COUNT(*) FROM timeline_events WHERE run_id IN ({placeholders})", run_ids).fetchone()[0])
            if not dry_run and run_ids:
                conn.executemany("DELETE FROM timeline_runs WHERE run_id = ?", [(rid,) for rid in run_ids])
                conn.commit()
            if not dry_run and vacuum:
                conn.execute("VACUUM")
            after_bytes = _db_file_bytes()
            return {
                "dry_run": dry_run,
                "days": days,
                "cutoff": cutoff,
                "cutoff_iso": _iso(cutoff),
                "runs_deleted": len(run_ids) if not dry_run else 0,
                "events_deleted": event_count if not dry_run else 0,
                "runs_matched": len(run_ids),
                "events_matched": event_count,
                "vacuumed": bool(vacuum and not dry_run),
                "bytes_before": before_bytes,
                "bytes_after": after_bytes,
            }
        finally:
            conn.close()


def vacuum_timeline() -> dict[str, Any]:
    before_bytes = _db_file_bytes()
    with _DB_LOCK:
        conn = _ensure_db()
        try:
            conn.execute("VACUUM")
            after_bytes = _db_file_bytes()
            return {"db_path": str(_db_path()), "bytes_before": before_bytes, "bytes_after": after_bytes}
        finally:
            conn.close()


def _preset_path() -> Path:
    base = get_hermes_home() / "timelines"
    base.mkdir(parents=True, exist_ok=True)
    return base / "presets.json"


def load_presets() -> dict[str, dict[str, Any]]:
    path = _preset_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    presets = data.get("presets", data) if isinstance(data, dict) else {}
    if not isinstance(presets, dict):
        return {}
    return {str(k): dict(v) for k, v in presets.items() if isinstance(v, dict)}


def save_preset(name: str, *, platform: str = "", chat_id: str = "", thread_id: str = "", source: str = "", limit: int | None = None) -> dict[str, Any]:
    if not name:
        raise ValueError("preset name is required")
    if not (platform or chat_id or source):
        raise ValueError("preset requires at least platform/chat_id or source")
    presets = load_presets()
    old = presets.get(name, {})
    now = _iso()
    preset = {
        "name": name,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "source": source,
        "limit": int(limit) if limit else old.get("limit") or 50,
        "created_at": old.get("created_at") or now,
        "updated_at": now,
    }
    presets[name] = preset
    path = _preset_path()
    path.write_text(json.dumps({"presets": presets}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return preset


def get_preset(name: str) -> dict[str, Any]:
    presets = load_presets()
    preset = presets.get(name)
    if not preset:
        raise SystemExit(f"Timeline preset not found: {name}")
    return preset


def _apply_preset_args(args: argparse.Namespace) -> argparse.Namespace:
    name = getattr(args, "preset", "") or ""
    if not name:
        return args
    preset = get_preset(name)
    for key in ("platform", "source", "chat_id", "thread_id"):
        if hasattr(args, key) and not getattr(args, key):
            setattr(args, key, preset.get(key) or "")
    if hasattr(args, "limit") and (getattr(args, "limit", None) in (None, 0)):
        setattr(args, "limit", int(preset.get("limit") or 50))
    return args


def _format_bytes(num: Any) -> str:
    try:
        value = float(num or 0)
    except Exception:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def _format_ms(ms: Any) -> str:
    if ms is None:
        return ""
    try:
        ms = int(ms)
    except Exception:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.2f}s"


def _print_list(args: argparse.Namespace) -> None:
    rows = list_runs(
        limit=args.limit,
        status=args.status or "",
        session_id=args.session_id or "",
        platform=args.platform or "",
        source=args.source or "",
    )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return
    if not rows:
        print("No timeline runs found.")
        return
    for r in rows:
        started = _iso(float(r["started_at"])) if r.get("started_at") else ""
        print(f"{r['run_id']}  {r.get('status','')}  {_format_ms(r.get('duration_ms'))}  {started}")
        meta = " ".join(x for x in [r.get("platform") or "", r.get("model") or "", r.get("summary") or ""] if x)
        if meta:
            print(f"  {meta}")


def _print_show(args: argparse.Namespace) -> None:
    run, events = get_run(args.run_id)
    if not run:
        raise SystemExit(f"Timeline run not found: {args.run_id}")
    if args.json:
        print(json.dumps({"run": run, "events": events}, ensure_ascii=False, indent=2, default=str))
        return
    started = float(run["started_at"])
    print(f"Run: {run['run_id']}")
    print(f"Status: {run.get('status')}  Duration: {_format_ms(run.get('duration_ms'))}  Started: {_iso(started)}")
    bits = [f"session={run.get('session_id') or '-'}", f"turn={run.get('turn_id') or '-'}", f"platform={run.get('platform') or '-'}", f"model={run.get('model') or '-'}"]
    print(" ".join(bits))
    if run.get("summary"):
        print(f"Summary: {run['summary']}")
    print("")
    for e in events:
        rel = float(e["ts"]) - started
        dur = f" ({_format_ms(e.get('duration_ms'))})" if e.get("duration_ms") is not None else ""
        status = f" [{e.get('status')}]" if e.get("status") else ""
        name = e.get("name") or e.get("event_type")
        print(f"{rel:8.3f}s  {e['event_type']:<14} {name}{status}{dur}")
        if e.get("summary"):
            print(f"           {e['summary']}")
        if args.payload:
            try:
                payload = json.loads(e.get("payload_json") or "{}")
            except Exception:
                payload = e.get("payload_json")
            print("           payload=" + json.dumps(payload, ensure_ascii=False, default=str)[:2000])


def _print_thread(args: argparse.Namespace) -> None:
    args = _apply_preset_args(args)
    if not (args.platform and args.chat_id):
        raise SystemExit("timeline thread requires --platform/--chat-id or --preset <name>")
    if not getattr(args, "limit", 0):
        args.limit = 20
    rows = list_thread_runs(
        platform=args.platform or "",
        chat_id=args.chat_id or "",
        thread_id=args.thread_id or "",
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return
    for row in rows:
        print(f"{row['run_id']}  {row['status']}  {_format_ms(row.get('duration_ms'))}  {_iso(row['started_at'])}")
        print(f"  {row.get('platform') or ''} {row.get('source') or ''} {row.get('summary') or ''}".rstrip())


def _print_dashboard(args: argparse.Namespace) -> None:
    from .dashboard import write_dashboard

    args = _apply_preset_args(args)
    if not getattr(args, "limit", 0):
        args.limit = 50
    output = args.output or str(get_hermes_home() / "timelines" / "dashboard.html")
    out = write_dashboard(
        output,
        list_runs=list_runs,
        list_thread_runs=list_thread_runs,
        get_run=get_run,
        iso=_iso,
        limit=args.limit,
        platform=args.platform or "",
        source=args.source or "",
        chat_id=args.chat_id or "",
        thread_id=args.thread_id or "",
    )
    print(str(out))


def _print_serve(args: argparse.Namespace) -> None:
    from .server import serve_dashboard

    args = _apply_preset_args(args)
    if not getattr(args, "limit", 0):
        args.limit = 50
    serve_dashboard(
        list_runs=list_runs,
        list_thread_runs=list_thread_runs,
        get_run=get_run,
        iso=_iso,
        host=args.host,
        port=args.port,
        limit=args.limit,
        platform=args.platform or "",
        source=args.source or "",
        chat_id=args.chat_id or "",
        thread_id=args.thread_id or "",
        open_browser=args.open,
    )


def _print_current(args: argparse.Namespace) -> None:
    if not getattr(args, "preset", ""):
        args.preset = "current"
    for key in ("platform", "chat_id", "thread_id", "source"):
        if not hasattr(args, key):
            setattr(args, key, "")
    _print_thread(args)


def _print_preset(args: argparse.Namespace) -> None:
    action = args.preset_action
    if action in (None, "list", "ls"):
        presets = load_presets()
        if args.json:
            print(json.dumps(presets, ensure_ascii=False, indent=2, sort_keys=True, default=str))
            return
        if not presets:
            print("No timeline presets found.")
            return
        for name, preset in sorted(presets.items()):
            bits = [preset.get("platform") or "-", preset.get("chat_id") or "-", preset.get("thread_id") or "-"]
            print(f"{name}  {' '.join(bits)}")
        return
    if action == "show":
        preset = get_preset(args.name)
        if args.json:
            print(json.dumps(preset, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        else:
            for key in ("name", "platform", "chat_id", "thread_id", "source", "limit", "updated_at"):
                print(f"{key}: {preset.get(key) or ''}")
        return
    if action == "save":
        preset = save_preset(
            args.name,
            platform=args.platform or "",
            chat_id=args.chat_id or "",
            thread_id=args.thread_id or "",
            source=args.source or "",
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(preset, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        else:
            print(f"Saved timeline preset '{args.name}'")
            print(f"  platform={preset.get('platform') or '-'} chat_id={preset.get('chat_id') or '-'} thread_id={preset.get('thread_id') or '-'}")
        return
    raise SystemExit(f"Unknown preset action: {action}")


def _print_stats(args: argparse.Namespace) -> None:
    stats = timeline_stats()
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
        return
    print(f"Timeline DB: {stats['db_path']}")
    print(f"Size: {_format_bytes(stats['db_bytes'])}")
    print(f"Runs: {stats['runs']}  Events: {stats['events']}")
    if stats.get("oldest_started_at_iso"):
        print(f"Oldest: {stats['oldest_started_at_iso']}")
    if stats.get("newest_started_at_iso"):
        print(f"Newest: {stats['newest_started_at_iso']}")
    if stats.get("statuses"):
        print("Statuses: " + ", ".join(f"{k or '-'}={v}" for k, v in sorted(stats["statuses"].items())))
    if stats.get("platforms"):
        print("Platforms: " + ", ".join(f"{k or '-'}={v}" for k, v in sorted(stats["platforms"].items())))


def _print_prune(args: argparse.Namespace) -> None:
    result = prune_timeline(days=args.days, dry_run=args.dry_run, vacuum=args.vacuum)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    action = "Would delete" if result["dry_run"] else "Deleted"
    runs = result["runs_matched"] if result["dry_run"] else result["runs_deleted"]
    events = result["events_matched"] if result["dry_run"] else result["events_deleted"]
    print(f"{action} {runs} runs and {events} events older than {args.days} days (before {result['cutoff_iso']}).")
    if result.get("vacuumed"):
        print(f"Vacuumed: {_format_bytes(result['bytes_before'])} -> {_format_bytes(result['bytes_after'])}")
    elif args.dry_run:
        print("Dry run only; re-run without --dry-run to delete.")


def _print_vacuum(args: argparse.Namespace) -> None:
    result = vacuum_timeline()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    print(f"Vacuumed {result['db_path']}: {_format_bytes(result['bytes_before'])} -> {_format_bytes(result['bytes_after'])}")


def _print_default(args: argparse.Namespace) -> None:
    for name, value in {
        "limit": 20,
        "status": "",
        "session_id": "",
        "platform": "",
        "source": "",
        "json": False,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    _print_list(args)


def setup_cli(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="timeline_action")
    p_list = sub.add_parser("list", aliases=["ls"], help="List recent local timeline runs")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--status", default="")
    p_list.add_argument("--session-id", default="")
    p_list.add_argument("--platform", default="", help="Filter runs by platform, e.g. slack or cli")
    p_list.add_argument("--source", default="", help="Filter runs by source prefix, e.g. slack:C123:178...")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_print_list)

    p_show = sub.add_parser("show", help="Show a local timeline run")
    p_show.add_argument("run_id")
    p_show.add_argument("--payload", action="store_true", help="Include redacted event payload previews")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=_print_show)

    p_thread = sub.add_parser("thread", help="List runs for a platform chat/thread")
    p_thread.add_argument("--preset", default="", help="Use a saved route preset, e.g. current")
    p_thread.add_argument("--platform", default="", help="Platform name, e.g. slack")
    p_thread.add_argument("--chat-id", default="", help="Platform chat/channel id")
    p_thread.add_argument("--thread-id", default="", help="Optional platform thread/topic id")
    p_thread.add_argument("--limit", type=int, default=0)
    p_thread.add_argument("--json", action="store_true")
    p_thread.set_defaults(func=_print_thread)

    p_current = sub.add_parser("current", help="Shortcut for `thread --preset current`")
    p_current.add_argument("--preset", default="current", help="Preset name to use; defaults to current")
    p_current.add_argument("--limit", type=int, default=0)
    p_current.add_argument("--json", action="store_true")
    p_current.set_defaults(func=_print_current)

    p_preset = sub.add_parser("preset", aliases=["presets"], help="Manage saved timeline route presets")
    preset_sub = p_preset.add_subparsers(dest="preset_action")
    p_preset.set_defaults(func=_print_preset, preset_action="list", json=False)
    p_preset_list = preset_sub.add_parser("list", aliases=["ls"], help="List saved presets")
    p_preset_list.add_argument("--json", action="store_true")
    p_preset_list.set_defaults(func=_print_preset, preset_action="list")
    p_preset_show = preset_sub.add_parser("show", help="Show a saved preset")
    p_preset_show.add_argument("name")
    p_preset_show.add_argument("--json", action="store_true")
    p_preset_show.set_defaults(func=_print_preset, preset_action="show")
    p_preset_save = preset_sub.add_parser("save", help="Save a platform chat/thread route preset")
    p_preset_save.add_argument("name")
    p_preset_save.add_argument("--platform", default="", help="Platform name, e.g. slack")
    p_preset_save.add_argument("--chat-id", default="", help="Platform chat/channel id")
    p_preset_save.add_argument("--thread-id", default="", help="Optional platform thread/topic id")
    p_preset_save.add_argument("--source", default="", help="Optional source prefix")
    p_preset_save.add_argument("--limit", type=int, default=50)
    p_preset_save.add_argument("--json", action="store_true")
    p_preset_save.set_defaults(func=_print_preset, preset_action="save")

    p_dash = sub.add_parser("dashboard", aliases=["html"], help="Write a self-contained local HTML timeline dashboard")
    p_dash.add_argument("--preset", default="", help="Use a saved route preset, e.g. current")
    p_dash.add_argument("--output", "-o", default="", help="Output HTML path; defaults to ~/.hermes/timelines/dashboard.html")
    p_dash.add_argument("--limit", type=int, default=0)
    p_dash.add_argument("--platform", default="", help="Filter runs by platform, e.g. slack")
    p_dash.add_argument("--source", default="", help="Filter runs by source prefix")
    p_dash.add_argument("--chat-id", default="", help="Filter dashboard to a platform chat/channel id")
    p_dash.add_argument("--thread-id", default="", help="Filter dashboard to a platform thread/topic id")
    p_dash.set_defaults(func=_print_dashboard)

    p_serve = sub.add_parser("serve", help="Serve a live local timeline dashboard over HTTP")
    p_serve.add_argument("--preset", default="", help="Use a saved route preset, e.g. current")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host; defaults to local-only 127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--limit", type=int, default=0)
    p_serve.add_argument("--platform", default="", help="Filter runs by platform, e.g. slack")
    p_serve.add_argument("--source", default="", help="Filter runs by source prefix")
    p_serve.add_argument("--chat-id", default="", help="Filter dashboard to a platform chat/channel id")
    p_serve.add_argument("--thread-id", default="", help="Filter dashboard to a platform thread/topic id")
    p_serve.add_argument("--open", action="store_true", help="Open the dashboard in the default browser")
    p_serve.set_defaults(func=_print_serve)

    p_stats = sub.add_parser("stats", help="Show local timeline database size and row counts")
    p_stats.add_argument("--json", action="store_true")
    p_stats.set_defaults(func=_print_stats)

    p_prune = sub.add_parser("prune", help="Delete timeline runs older than N days")
    p_prune.add_argument("--days", type=int, required=True, help="Delete runs older than this many days")
    p_prune.add_argument("--dry-run", action="store_true", help="Report what would be deleted without changing the database")
    p_prune.add_argument("--vacuum", action="store_true", help="Run VACUUM after deleting rows")
    p_prune.add_argument("--json", action="store_true")
    p_prune.set_defaults(func=_print_prune)

    p_vacuum = sub.add_parser("vacuum", help="Compact the local timeline SQLite database")
    p_vacuum.add_argument("--json", action="store_true")
    p_vacuum.set_defaults(func=_print_vacuum)

    parser.set_defaults(func=_print_default)


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
    ctx.register_hook("post_gateway_delivery", on_post_gateway_delivery)
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_turn)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_cli_command(
        "timeline",
        help="Inspect local Hermes Agent Timeline runs",
        description="List and show local agent timeline events stored by the observability/timeline plugin.",
        setup_fn=setup_cli,
    )
