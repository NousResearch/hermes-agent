"""session_continuity plugin.

Injects a small, deterministic continuity block on the first turn of a fresh
user-facing session. The hook returns ``{"context": ...}``, so Hermes appends
the block to the current user message at API-call time; the cached system prompt
is never changed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 2000
DEFAULT_MESSAGE_LIMIT = 6
MIN_MAX_CHARS = 500
MAX_MAX_CHARS = 8000
MIN_MESSAGE_LIMIT = 2
MAX_MESSAGE_LIMIT = 12

EXCLUDED_SOURCES = {"subagent", "tool", "cron", "curator", "catalog"}
EXCLUDED_PLATFORMS = EXCLUDED_SOURCES | {"background_review"}


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _load_settings() -> Dict[str, int]:
    """Read optional plugin settings from config.yaml, fail-closed."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        section = cfg.get("session_continuity", {})
        if not isinstance(section, dict):
            section = {}
        return {
            "max_chars": _clamp_int(
                section.get("max_chars"),
                DEFAULT_MAX_CHARS,
                MIN_MAX_CHARS,
                MAX_MAX_CHARS,
            ),
            "message_limit": _clamp_int(
                section.get("message_limit"),
                DEFAULT_MESSAGE_LIMIT,
                MIN_MESSAGE_LIMIT,
                MAX_MESSAGE_LIMIT,
            ),
        }
    except Exception:
        return {
            "max_chars": DEFAULT_MAX_CHARS,
            "message_limit": DEFAULT_MESSAGE_LIMIT,
        }


def _open_session_db():
    try:
        from hermes_state import SessionDB

        return SessionDB()
    except Exception as exc:
        logger.debug("session_continuity: SessionDB unavailable: %s", exc)
        return None


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _format_ts(ts: Any) -> str:
    try:
        value = float(ts)
    except (TypeError, ValueError):
        return ""
    try:
        return (
            datetime.fromtimestamp(value, timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except Exception:
        return ""


def _has_prior_dialog(
    conversation_history: Optional[Iterable[Dict[str, Any]]],
    user_message: Any,
) -> bool:
    """Return True if the hook payload already contains prior chat turns."""
    if not conversation_history:
        return False
    dialog = [
        m for m in conversation_history
        if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
    ]
    if not dialog:
        return False
    if len(dialog) == 1 and dialog[0].get("role") == "user":
        current = _normalize_text(user_message)
        seen = _normalize_text(dialog[0].get("content"))
        return bool(seen and current and seen != current)
    return True


def _is_user_facing_platform(platform: Any) -> bool:
    value = str(platform or "").strip().lower()
    if not value:
        return True
    return value not in EXCLUDED_PLATFORMS


def _source_filter_for_platform(platform: Any) -> str:
    """Return the session source to recall from for a hook platform value.

    Continuity is intentionally scoped to the same surface by default. Without
    this, enabling the plugin in a gateway could leak a private CLI/Desktop
    session into a group chat's first turn. Empty platform values mirror
    Hermes' session creation fallback and are treated as ``cli``.
    """
    value = str(platform or "").strip().lower()
    if value in EXCLUDED_PLATFORMS:
        return ""
    return value or "cli"


def _recent_visible_session(
    db,
    current_session_id: str,
    *,
    source: str = "",
    user_id: str = "",
) -> Optional[Dict[str, Any]]:
    try:
        rows = db.list_sessions_rich(
            source=source or None,
            exclude_sources=sorted(EXCLUDED_SOURCES),
            limit=20,
            include_children=False,
            min_message_count=1,
            order_by_last_active=True,
            include_archived=False,
        )
    except Exception as exc:
        logger.debug("session_continuity: failed to list sessions: %s", exc)
        return None

    current = str(current_session_id or "")
    current_user = str(user_id or "")
    for row in rows:
        sid = str(row.get("id") or "")
        if not sid or sid == current:
            continue
        row_source = str(row.get("source") or "").lower()
        if row_source in EXCLUDED_SOURCES:
            continue
        row_user = str(row.get("user_id") or "")
        if current_user and row_user and row_user != current_user:
            continue
        if int(row.get("message_count") or 0) <= 0:
            continue
        return row
    return None


def _recent_dialog_messages(db, session_id: str, limit: int) -> List[Dict[str, str]]:
    try:
        rows = db.get_messages(session_id)
    except Exception as exc:
        logger.debug("session_continuity: failed to load messages: %s", exc)
        return []

    dialog: List[Dict[str, str]] = []
    for row in rows:
        role = row.get("role")
        if role not in {"user", "assistant"}:
            continue
        if row.get("tool_call_id") or row.get("tool_name"):
            continue
        if row.get("tool_calls") and not row.get("content"):
            continue
        content = _normalize_text(row.get("content"))
        if not content:
            continue
        dialog.append({"role": str(role), "content": content})
    return dialog[-limit:]


def _build_block(
    session: Dict[str, Any],
    messages: List[Dict[str, str]],
    *,
    max_chars: int,
) -> Optional[str]:
    sid = str(session.get("id") or "")
    if not sid or not messages:
        return None

    title = _normalize_text(session.get("title")) or "(untitled)"
    source = _normalize_text(session.get("source")) or "unknown"
    last_active = _format_ts(session.get("last_active") or session.get("started_at"))
    sid_prefix = sid[:8]

    header = [
        "[Session continuity: most recent previous visible session]",
        f"Session: {sid_prefix}",
        f"Source: {source}",
        f"Title: {title}",
    ]
    if last_active:
        header.append(f"Last active: {last_active}")
    header.append("Recent dialog:")

    budget_for_messages = max(120, max_chars - sum(len(line) + 1 for line in header))
    per_message = max(80, budget_for_messages // max(1, len(messages)))
    lines = list(header)
    for msg in messages:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"- {label}: {_clip(msg['content'], per_message)}")

    block = "\n".join(lines)
    return _clip(block, max_chars)


def build_continuity_context(
    *,
    session_db: Any = None,
    current_session_id: str = "",
    is_first_turn: bool = True,
    conversation_history: Optional[Iterable[Dict[str, Any]]] = None,
    user_message: Any = "",
    platform: Any = "",
    sender_id: Any = "",
    max_chars: Optional[int] = None,
    message_limit: Optional[int] = None,
) -> Optional[str]:
    """Return the continuity block, or ``None`` when nothing should inject."""
    if not is_first_turn:
        return None
    if not _is_user_facing_platform(platform):
        return None
    if _has_prior_dialog(conversation_history, user_message):
        return None

    settings = _load_settings()
    max_chars = _clamp_int(
        max_chars if max_chars is not None else settings["max_chars"],
        DEFAULT_MAX_CHARS,
        MIN_MAX_CHARS,
        MAX_MAX_CHARS,
    )
    message_limit = _clamp_int(
        message_limit if message_limit is not None else settings["message_limit"],
        DEFAULT_MESSAGE_LIMIT,
        MIN_MESSAGE_LIMIT,
        MAX_MESSAGE_LIMIT,
    )

    db = session_db if session_db is not None else _open_session_db()
    if db is None:
        return None

    session = _recent_visible_session(
        db,
        current_session_id,
        source=_source_filter_for_platform(platform),
        user_id=str(sender_id or ""),
    )
    if not session:
        return None
    messages = _recent_dialog_messages(db, str(session["id"]), message_limit)
    return _build_block(session, messages, max_chars=max_chars)


def pre_llm_call(
    session_id: str = "",
    is_first_turn: bool = False,
    conversation_history: Optional[Iterable[Dict[str, Any]]] = None,
    user_message: Any = "",
    platform: Any = "",
    sender_id: Any = "",
    session_db: Any = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Hook callback registered with Hermes."""
    try:
        context = build_continuity_context(
            current_session_id=session_id,
            is_first_turn=is_first_turn,
            conversation_history=conversation_history,
            user_message=user_message,
            platform=platform,
            sender_id=sender_id,
            session_db=session_db,
        )
        if context:
            return {"context": context}
    except Exception as exc:
        logger.debug("session_continuity: hook failed open: %s", exc)
    return None


def continuity_status(*, session_db: Any = None, current_session_id: str = "") -> str:
    """Small testable status helper used by the slash command."""
    db = session_db if session_db is not None else _open_session_db()
    if db is None:
        return "Session continuity is enabled, but SessionDB is unavailable."
    session = _recent_visible_session(db, current_session_id)
    if not session:
        return "Session continuity is enabled. No previous visible session is available."
    sid = str(session.get("id") or "")
    source = _normalize_text(session.get("source")) or "unknown"
    title = _normalize_text(session.get("title")) or "(untitled)"
    return (
        "Session continuity is enabled. "
        f"Next fresh session can recall {sid[:8]} ({source}) - {title}."
    )


def _handle_continuity_command(raw_args: str = "") -> str:
    sub = (raw_args or "").strip().split(maxsplit=1)[0].lower()
    if sub in {"", "status"}:
        return continuity_status()
    if sub == "reset":
        return "Session continuity has no persistent plugin state to reset."
    return "Usage: /continuity [status|reset]"


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_command(
        "continuity",
        handler=_handle_continuity_command,
        description="Show session-continuity plugin status.",
        args_hint="[status|reset]",
    )
