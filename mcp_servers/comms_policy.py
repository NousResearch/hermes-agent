"""Telegram / Comms Policy MCP server.

This stdio MCP server centralizes Brandon-style concise notification formatting,
SQLite-backed dedupe state, and stale-alert suppression for Hermes automations.
It intentionally delegates actual delivery to the existing ``send_message`` tool
implementation so credentials and platform routing stay in one place.

Usage:
    python -m mcp_servers.comms_policy

Hermes config snippet:
    mcp_servers:
      comms_policy:
        command: python
        args: ["-m", "mcp_servers.comms_policy"]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("hermes.mcp_servers.comms_policy")

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_SERVER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by tests via monkeypatchable fallback
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_SERVER_AVAILABLE = False

DEFAULT_DEDUPE_WINDOW_SECONDS = 60 * 60
DEFAULT_STALE_AFTER_SECONDS = 60 * 30
MAX_MESSAGE_CHARS = 1600


@dataclass(frozen=True)
class PolicyResult:
    """Result from policy evaluation before delivery."""

    allowed: bool
    reason: str
    dedupe_key: str
    message_hash: str
    last_sent_at: Optional[float] = None
    age_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "dedupe_key": self.dedupe_key,
            "message_hash": self.message_hash,
            "last_sent_at": self.last_sent_at,
            "age_seconds": self.age_seconds,
        }


def _get_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()


def default_db_path() -> Path:
    """Return the profile-scoped SQLite DB path for comms policy state."""

    return _get_hermes_home() / "comms_policy" / "notifications.sqlite"


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL,
            category TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            message_hash TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            event_ts REAL,
            queued_at REAL NOT NULL,
            sent_at REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notifications_lookup
        ON notifications (target, dedupe_key, message_hash, sent_at DESC, queued_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notifications_queue
        ON notifications (status, queued_at)
        """
    )
    conn.commit()


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _message_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _stable_key(*parts: Any) -> str:
    normalized = "|".join(_normalize_text(part).lower() for part in parts if part is not None)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _coerce_int(value: Any, *, default: int, minimum: int = 0, maximum: int = 30 * 24 * 3600) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(coerced, maximum))


def _clip(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _bullet_lines(items: Any, *, max_items: int = 4) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        raw_items = [line.strip(" -•\t") for line in items.splitlines() if line.strip()]
    elif isinstance(items, (list, tuple)):
        raw_items = [_normalize_text(item) for item in items if _normalize_text(item)]
    else:
        raw_items = [_normalize_text(items)]
    return [f"• {item}" for item in raw_items[:max_items] if item]


def format_success_notice(
    title: str,
    details: Any = None,
    *,
    task_id: str = "",
    next_step: str = "",
) -> str:
    """Format a concise Telegram success notice."""

    lines = [f"✅ {_normalize_text(title) or 'Done'}"]
    if task_id:
        lines.append(f"Task: {task_id}")
    lines.extend(_bullet_lines(details))
    if next_step:
        lines.append(f"Next: {_normalize_text(next_step)}")
    return _clip("\n".join(lines))


def format_failure_notice(
    title: str,
    error: Any = None,
    *,
    task_id: str = "",
    next_step: str = "",
) -> str:
    """Format a concise Telegram failure / Kanban error notice."""

    lines = [f"❌ {_normalize_text(title) or 'Failed'}"]
    if task_id:
        lines.append(f"Task: {task_id}")
    if error:
        lines.append(f"Error: {_normalize_text(error)}")
    if next_step:
        lines.append(f"Next: {_normalize_text(next_step)}")
    return _clip("\n".join(lines))


def format_briefing(
    title: str,
    items: Any = None,
    *,
    footer: str = "",
) -> str:
    """Format a concise briefing for Telegram."""

    lines = [f"📋 {_normalize_text(title) or 'Briefing'}"]
    lines.extend(_bullet_lines(items, max_items=8))
    if footer:
        lines.append(_normalize_text(footer))
    return _clip("\n".join(lines))


def format_ready_question(question: str, *, context: Any = None) -> str:
    """Format a ready-question convention for Kanban/human decisions."""

    lines = [f"❓ {_normalize_text(question) or 'Decision needed'}"]
    lines.extend(_bullet_lines(context, max_items=3))
    lines.append("Reply with the choice or details.")
    return _clip("\n".join(lines))


def evaluate_policy(
    *,
    target: str,
    category: str,
    message: str,
    dedupe_key: str = "",
    dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
    event_ts: Optional[float] = None,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    now: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> PolicyResult:
    """Return whether a notification should be sent under dedupe/stale policy."""

    now_ts = time.time() if now is None else float(now)
    target = _normalize_text(target) or "telegram"
    category = _normalize_text(category) or "notice"
    message_hash = _message_hash(message)
    dedupe_key = _normalize_text(dedupe_key) or _stable_key(target, category, message)
    dedupe_window_seconds = _coerce_int(
        dedupe_window_seconds,
        default=DEFAULT_DEDUPE_WINDOW_SECONDS,
        maximum=30 * 24 * 3600,
    )
    stale_after_seconds = _coerce_int(
        stale_after_seconds,
        default=DEFAULT_STALE_AFTER_SECONDS,
        maximum=30 * 24 * 3600,
    )

    age_seconds = None
    if event_ts is not None and stale_after_seconds > 0:
        age_seconds = max(0.0, now_ts - float(event_ts))
        if age_seconds > stale_after_seconds:
            return PolicyResult(
                allowed=False,
                reason="stale",
                dedupe_key=dedupe_key,
                message_hash=message_hash,
                age_seconds=age_seconds,
            )

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT sent_at, queued_at FROM notifications
            WHERE target = ? AND dedupe_key = ? AND message_hash = ?
              AND status IN ('sent', 'queued')
            ORDER BY COALESCE(sent_at, queued_at) DESC
            LIMIT 1
            """,
            (target, dedupe_key, message_hash),
        ).fetchone()

    if row:
        last = row["sent_at"] if row["sent_at"] is not None else row["queued_at"]
        if last is not None and now_ts - float(last) < dedupe_window_seconds:
            return PolicyResult(
                allowed=False,
                reason="duplicate",
                dedupe_key=dedupe_key,
                message_hash=message_hash,
                last_sent_at=float(last),
                age_seconds=age_seconds,
            )

    return PolicyResult(
        allowed=True,
        reason="allowed",
        dedupe_key=dedupe_key,
        message_hash=message_hash,
        age_seconds=age_seconds,
    )


def record_notification(
    *,
    target: str,
    category: str,
    dedupe_key: str,
    message_hash: str,
    message: str,
    status: str,
    reason: str = "",
    event_ts: Optional[float] = None,
    now: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> int:
    now_ts = time.time() if now is None else float(now)
    sent_at = now_ts if status == "sent" else None
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO notifications (
                target, category, dedupe_key, message_hash, message, status,
                reason, event_ts, queued_at, sent_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target,
                category,
                dedupe_key,
                message_hash,
                message,
                status,
                reason,
                event_ts,
                now_ts,
                sent_at,
                _json(metadata or {}),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def check_last_sent_record(
    *,
    target: str = "telegram",
    message: str = "",
    dedupe_key: str = "",
    category: str = "notice",
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    target = _normalize_text(target) or "telegram"
    category = _normalize_text(category) or "notice"
    if not dedupe_key:
        dedupe_key = _stable_key(target, category, message) if message else ""
    msg_hash = _message_hash(message) if message else ""
    where = ["target = ?"]
    args: list[Any] = [target]
    if dedupe_key:
        where.append("dedupe_key = ?")
        args.append(dedupe_key)
    if msg_hash:
        where.append("message_hash = ?")
        args.append(msg_hash)
    where.append("status IN ('sent', 'queued')")
    with _connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT * FROM notifications
            WHERE {' AND '.join(where)}
            ORDER BY COALESCE(sent_at, queued_at) DESC
            LIMIT 1
            """,
            args,
        ).fetchone()
    if not row:
        return {"found": False, "target": target, "dedupe_key": dedupe_key}
    data = dict(row)
    data["found"] = True
    try:
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
    except json.JSONDecodeError:
        data["metadata"] = {}
    return data


def dedupe_notification_policy(
    *,
    target: str = "telegram",
    message: str,
    dedupe_key: str = "",
    category: str = "notice",
    dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
    event_ts: Optional[float] = None,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    now: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    result = evaluate_policy(
        target=target,
        category=category,
        message=message,
        dedupe_key=dedupe_key,
        dedupe_window_seconds=dedupe_window_seconds,
        event_ts=event_ts,
        stale_after_seconds=stale_after_seconds,
        now=now,
        db_path=db_path,
    )
    if not result.allowed:
        notification_id = record_notification(
            target=_normalize_text(target) or "telegram",
            category=_normalize_text(category) or "notice",
            dedupe_key=result.dedupe_key,
            message_hash=result.message_hash,
            message=message,
            status="suppressed",
            reason=result.reason,
            event_ts=event_ts,
            now=now,
            db_path=db_path,
        )
    else:
        notification_id = None
    payload = result.to_dict()
    payload.update(
        {
            "should_send": result.allowed,
            "suppressed": not result.allowed,
            "notification_id": notification_id,
        }
    )
    return payload


def _send_via_existing_tool(target: str, message: str) -> dict[str, Any]:
    """Deliver through Hermes' existing send_message tool to avoid secret duplication."""

    try:
        from tools.send_message_tool import send_message_tool

        raw = send_message_tool({"action": "send", "target": target, "message": message})
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    except Exception as exc:
        logger.exception("send_message delivery failed")
        return {"error": str(exc)}


def send_policy_notice(
    *,
    target: str = "telegram",
    category: str,
    message: str,
    dedupe_key: str = "",
    dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
    event_ts: Optional[float] = None,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    dry_run: bool = False,
    now: Optional[float] = None,
    sender: Optional[Callable[[str, str], dict[str, Any]]] = None,
    metadata: Optional[dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    target = _normalize_text(target) or "telegram"
    category = _normalize_text(category) or "notice"
    policy = evaluate_policy(
        target=target,
        category=category,
        message=message,
        dedupe_key=dedupe_key,
        dedupe_window_seconds=dedupe_window_seconds,
        event_ts=event_ts,
        stale_after_seconds=stale_after_seconds,
        now=now,
        db_path=db_path,
    )
    if not policy.allowed:
        notification_id = record_notification(
            target=target,
            category=category,
            dedupe_key=policy.dedupe_key,
            message_hash=policy.message_hash,
            message=message,
            status="suppressed",
            reason=policy.reason,
            event_ts=event_ts,
            now=now,
            metadata=metadata,
            db_path=db_path,
        )
        payload = policy.to_dict()
        payload.update({"sent": False, "suppressed": True, "notification_id": notification_id, "message": message})
        return payload

    delivery: dict[str, Any] = {"dry_run": True} if dry_run else (sender or _send_via_existing_tool)(target, message)
    status = "sent" if "error" not in delivery and not dry_run else "queued" if dry_run else "failed"
    reason = "dry_run" if dry_run else str(delivery.get("error", ""))
    notification_id = record_notification(
        target=target,
        category=category,
        dedupe_key=policy.dedupe_key,
        message_hash=policy.message_hash,
        message=message,
        status=status,
        reason=reason,
        event_ts=event_ts,
        now=now,
        metadata=metadata,
        db_path=db_path,
    )
    payload = policy.to_dict()
    payload.update(
        {
            "sent": status == "sent",
            "queued": status == "queued",
            "suppressed": False,
            "notification_id": notification_id,
            "message": message,
            "delivery": delivery,
        }
    )
    return payload


def build_server() -> Any:
    """Build and return the FastMCP server instance."""

    if not _MCP_SERVER_AVAILABLE or FastMCP is None:
        raise RuntimeError("MCP SDK not available. Install with: pip install mcp")

    mcp = FastMCP("hermes-comms-policy")

    @mcp.tool()
    def send_success_notice(
        title: str,
        details: Any = None,
        target: str = "telegram",
        task_id: str = "",
        next_step: str = "",
        dedupe_key: str = "",
        dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
        event_ts: Optional[float] = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
        dry_run: bool = False,
    ) -> str:
        """Format, dedupe, and send a concise success notice."""

        message = format_success_notice(title, details, task_id=task_id, next_step=next_step)
        return _json(
            send_policy_notice(
                target=target,
                category="success",
                message=message,
                dedupe_key=dedupe_key or _stable_key("success", task_id, title, message),
                dedupe_window_seconds=dedupe_window_seconds,
                event_ts=event_ts,
                stale_after_seconds=stale_after_seconds,
                dry_run=dry_run,
                metadata={"title": title, "task_id": task_id},
            )
        )

    @mcp.tool()
    def send_failure_notice(
        title: str,
        error: Any = None,
        target: str = "telegram",
        task_id: str = "",
        next_step: str = "",
        dedupe_key: str = "",
        dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
        event_ts: Optional[float] = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
        dry_run: bool = False,
    ) -> str:
        """Format, dedupe, and send a concise failure / Kanban error notice."""

        message = format_failure_notice(title, error, task_id=task_id, next_step=next_step)
        return _json(
            send_policy_notice(
                target=target,
                category="failure",
                message=message,
                dedupe_key=dedupe_key or _stable_key("failure", task_id, title, error),
                dedupe_window_seconds=dedupe_window_seconds,
                event_ts=event_ts,
                stale_after_seconds=stale_after_seconds,
                dry_run=dry_run,
                metadata={"title": title, "task_id": task_id},
            )
        )

    @mcp.tool()
    def send_briefing(
        title: str,
        items: Any = None,
        target: str = "telegram",
        footer: str = "",
        dedupe_key: str = "",
        dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
        event_ts: Optional[float] = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
        dry_run: bool = False,
    ) -> str:
        """Format, dedupe, and send a concise Telegram-style briefing."""

        message = format_briefing(title, items, footer=footer)
        return _json(
            send_policy_notice(
                target=target,
                category="briefing",
                message=message,
                dedupe_key=dedupe_key or _stable_key("briefing", title, message),
                dedupe_window_seconds=dedupe_window_seconds,
                event_ts=event_ts,
                stale_after_seconds=stale_after_seconds,
                dry_run=dry_run,
                metadata={"title": title},
            )
        )

    @mcp.tool()
    def queue_notification(
        message: str,
        target: str = "telegram",
        category: str = "notice",
        dedupe_key: str = "",
        dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
        event_ts: Optional[float] = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ) -> str:
        """Queue a notification after applying duplicate/stale suppression."""

        result = send_policy_notice(
            target=target,
            category=category,
            message=_clip(message),
            dedupe_key=dedupe_key,
            dedupe_window_seconds=dedupe_window_seconds,
            event_ts=event_ts,
            stale_after_seconds=stale_after_seconds,
            dry_run=True,
        )
        return _json(result)

    @mcp.tool()
    def dedupe_notification(
        message: str,
        target: str = "telegram",
        category: str = "notice",
        dedupe_key: str = "",
        dedupe_window_seconds: int = DEFAULT_DEDUPE_WINDOW_SECONDS,
        event_ts: Optional[float] = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ) -> str:
        """Check whether a notification is duplicate or stale before sending."""

        return _json(
            dedupe_notification_policy(
                target=target,
                category=category,
                message=_clip(message),
                dedupe_key=dedupe_key,
                dedupe_window_seconds=dedupe_window_seconds,
                event_ts=event_ts,
                stale_after_seconds=stale_after_seconds,
            )
        )

    @mcp.tool()
    def check_last_sent(
        target: str = "telegram",
        message: str = "",
        dedupe_key: str = "",
        category: str = "notice",
    ) -> str:
        """Return the latest stored notification matching target/key/message."""

        return _json(check_last_sent_record(target=target, message=message, dedupe_key=dedupe_key, category=category))

    @mcp.tool()
    def format_ready_question_notice(question: str, context: Any = None) -> str:
        """Format Brandon's concise ready-question convention without sending."""

        return _json({"message": format_ready_question(question, context=context)})

    return mcp


def main() -> int:
    logging.basicConfig(level=os.getenv("HERMES_COMMS_POLICY_LOG_LEVEL", "WARNING"))
    try:
        server = build_server()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
