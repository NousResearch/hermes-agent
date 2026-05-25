"""Prompt-only context bridge between automatically rotated gateway sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from agent.context_compressor import LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX
from agent.memory_manager import sanitize_context
from gateway.config import BridgeConfig, Platform
from gateway.session import SessionBridgeCandidate, SessionSource

_MAX_MESSAGE_CHARS = 700
_MAX_BRIDGE_CHARS = 10000


@dataclass
class SessionBridgeResult:
    text: Optional[str]
    applied: bool
    reason: str
    previous_session_id: Optional[str] = None


def build_session_bridge_context(
    *,
    session_db: Any,
    config: BridgeConfig,
    source: SessionSource,
    current_session_id: str,
    previous: Optional[SessionBridgeCandidate],
    now: Optional[datetime] = None,
) -> SessionBridgeResult:
    """Build a one-shot recovery block for a previous gateway session."""
    if not getattr(config, "enabled", True):
        return SessionBridgeResult(None, False, "disabled")
    if source.platform == Platform.LOCAL:
        return SessionBridgeResult(None, False, "local")
    if not session_db:
        return SessionBridgeResult(None, False, "no_db")
    if previous is None or not previous.session_id:
        return SessionBridgeResult(None, False, "no_previous")
    if previous.session_id == current_session_id:
        return SessionBridgeResult(None, False, "same_session")

    session = None
    try:
        session = session_db.get_session(previous.session_id)
    except Exception:
        session = None
    ended_at = previous.ended_at or _datetime_from_epoch(
        session.get("ended_at") if session else None
    )
    if ended_at is None:
        return SessionBridgeResult(None, False, "no_end_time", previous.session_id)

    now = now or datetime.now()
    age = now - ended_at
    if age < timedelta(0):
        age = timedelta(0)
    if age > timedelta(minutes=int(getattr(config, "max_age_minutes", 1440))):
        return SessionBridgeResult(None, False, "too_old", previous.session_id)

    try:
        messages = session_db.get_messages(previous.session_id)
    except Exception:
        return SessionBridgeResult(None, False, "messages_unavailable", previous.session_id)

    useful = [_normalize_message(m) for m in messages if _is_useful_message(m)]
    useful = [m for m in useful if m["content"]]
    if len(useful) < 2:
        return SessionBridgeResult(None, False, "too_few_messages", previous.session_id)

    summary = None
    if getattr(config, "include_summary", True):
        summary = _latest_compression_summary(messages)
    if not summary:
        summary = _bookend_summary(useful)

    recent = useful[-int(getattr(config, "max_messages", 15)) :]
    freshness = _format_freshness(
        age,
        threshold_minutes=int(getattr(config, "freshness_threshold_minutes", 30)),
    )
    exchanges = "\n".join(
        f"{idx}. {msg['speaker']}: {_truncate(msg['content'], _MAX_MESSAGE_CHARS)}"
        for idx, msg in enumerate(recent, 1)
    )

    text = (
        "[CONTEXTE DE REPRISE]\n"
        f"La session precedente avec cet interlocuteur s'est terminee {freshness}. "
        "Ceci est un rappel contextuel, pas un nouveau tour de conversation.\n\n"
        f"RESUME : {_truncate(summary, 2400)}\n\n"
        f"DERNIERS ECHANGES :\n{exchanges}\n\n"
        "Reponds au message actuel en priorite et en continuite si le sujet correspond. "
        "Ne rejoue pas les actions deja faites et ne mentionne pas ce contexte injecte a l'humain."
    )
    return SessionBridgeResult(
        _truncate(text, _MAX_BRIDGE_CHARS),
        True,
        "applied",
        previous.session_id,
    )


def _datetime_from_epoch(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value))
    except (TypeError, ValueError, OSError):
        return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _is_summary_text(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(SUMMARY_PREFIX) or stripped.startswith(LEGACY_SUMMARY_PREFIX)


def _strip_summary_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _is_useful_message(message: dict[str, Any]) -> bool:
    if message.get("role") not in {"user", "assistant"}:
        return False
    return not _is_summary_text(_content_text(message.get("content")))


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role")
    speaker = "Utilisateur" if role == "user" else "Assistant"
    content = sanitize_context(_content_text(message.get("content"))).strip()
    return {"speaker": speaker, "content": content}


def _latest_compression_summary(messages: list[dict[str, Any]]) -> Optional[str]:
    for message in reversed(messages):
        text = _content_text(message.get("content"))
        if _is_summary_text(text):
            return sanitize_context(_strip_summary_prefix(text)).strip()
    return None


def _bookend_summary(messages: list[dict[str, str]]) -> str:
    if len(messages) <= 6:
        selected = messages
    else:
        selected = messages[:3] + messages[-3:]
    return " / ".join(
        f"{msg['speaker']}: {_truncate(msg['content'], 220)}"
        for msg in selected
        if msg["content"]
    )


def _format_freshness(age: timedelta, *, threshold_minutes: int) -> str:
    total_minutes = max(0, int(age.total_seconds() // 60))
    if total_minutes < threshold_minutes:
        return "a l'instant"
    if total_minutes < 120:
        return f"il y a {total_minutes} minutes"
    total_hours = max(1, total_minutes // 60)
    if total_hours < 48:
        return f"il y a {total_hours} heures"
    return f"il y a {total_hours // 24} jours"


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
