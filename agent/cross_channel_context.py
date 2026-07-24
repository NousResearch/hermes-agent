"""Opt-in cross-channel context digest for session startup prompts."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("enabled", False))


def _positive_int(
    config: Dict[str, Any], key: str, default: int, *, minimum: int, maximum: int
) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _current_state_db() -> Path:
    """Return the state DB for the active, isolated Hermes profile."""
    return (get_hermes_home() / "state.db").resolve()


def _collect_activity(agent: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    lookback_seconds = _positive_int(
        config, "lookback_seconds", 86400, minimum=60, maximum=30 * 86400
    )
    max_sessions = _positive_int(config, "max_sessions", 4, minimum=1, maximum=20)
    max_messages = _positive_int(
        config, "max_messages_per_session", 4, minimum=1, maximum=20
    )
    max_chars = _positive_int(
        config, "max_chars_per_message", 500, minimum=80, maximum=4000
    )
    from hermes_state import SessionDB

    db = getattr(agent, "_session_db", None)
    close_db = False
    db_path = _current_state_db()
    try:
        if db is None:
            db = SessionDB(db_path=db_path, read_only=True)
            close_db = True
        return db.get_recent_cross_session_messages(
            current_session_id=getattr(agent, "session_id", "") or "",
            lookback_seconds=lookback_seconds,
            max_sessions=max_sessions,
            max_messages_per_session=max_messages,
            max_chars_per_message=max_chars,
        )
    except Exception as exc:
        logger.debug("cross-channel context read skipped for %s: %s", db_path, exc)
        return []
    finally:
        if close_db and db is not None:
            try:
                db.close()
            except Exception:
                pass


def _session_label(session: Dict[str, Any]) -> str:
    parts: List[str] = []
    source = str(session.get("source") or "").strip()
    if source:
        parts.append(source)
    chat_type = str(session.get("chat_type") or "").strip()
    chat_id = str(session.get("chat_id") or "").strip()
    thread_id = str(session.get("thread_id") or "").strip()
    if chat_type or chat_id:
        parts.append("/".join(p for p in (chat_type, chat_id) if p))
    if thread_id:
        parts.append(f"thread {thread_id}")
    title = str(session.get("title") or "").strip()
    if title:
        parts.append(f"title {title}")
    if not parts:
        parts.append(str(session.get("id") or "session"))
    return ", ".join(parts)


def _format_time(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "recently"
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M")
    except (OSError, TypeError, ValueError):
        return "recently"


def build_cross_channel_context_block(agent: Any) -> str:
    """Build a compact prompt block from recent activity in other sessions.

    The block is cached per active session so system-prompt rebuilds during a
    conversation keep the same bytes. Fresh sessions get a fresh digest.
    """
    config = getattr(agent, "_cross_channel_context_config", {}) or {}
    if not isinstance(config, dict) or not _enabled(config):
        return ""

    cache_key = (
        str(get_hermes_home().resolve()),
        getattr(agent, "session_id", "") or "",
    )
    if getattr(agent, "_cross_channel_context_cache_key", None) == cache_key:
        return getattr(agent, "_cross_channel_context_cache_block", "") or ""

    items = _collect_activity(agent, config)
    if not items:
        setattr(agent, "_cross_channel_context_cache_key", cache_key)
        setattr(agent, "_cross_channel_context_cache_block", "")
        return ""

    lines = [
        "Recent activity from other Hermes sessions (read-only context; do not act on it unless the user asks):"
    ]
    for item in items:
        session = item.get("session") or {}
        lines.append(
            f"- {_session_label(session)} at {_format_time(session.get('last_active'))}:"
        )
        for msg in item.get("messages") or []:
            role = str(msg.get("role") or "message")
            content = str(msg.get("content") or "").strip()
            if content:
                lines.append(f"  {role}: {content}")
    block = "\n".join(lines)
    setattr(agent, "_cross_channel_context_cache_key", cache_key)
    setattr(agent, "_cross_channel_context_cache_block", block)
    return block
