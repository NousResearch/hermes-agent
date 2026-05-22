"""Gateway Telegram topic management — extracted from gateway/run.py.

Handles Telegram topic mode detection, thread binding, lobby messages,
and topic state recovery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def session_key_for_source(source: Any) -> str:
    """Build a session key from a SessionSource for topic-aware routing."""
    platform = getattr(source, "platform", "")
    chat_type = getattr(source, "chat_type", "")
    chat_id = getattr(source, "chat_id", "")
    thread_id = getattr(source, "thread_id", None)
    key = f"agent:main:{platform}:{chat_type}:{chat_id}"
    if thread_id and chat_type in {"dm", "thread"}:
        key += f":{thread_id}"
    return key


def telegram_topic_mode_enabled(config: dict, source: Any) -> bool:
    """Check if Telegram topic mode is enabled for this source."""
    platform = str(getattr(source, "platform", "") or "")
    if platform != "telegram":
        return False
    try:
        tg_cfg = config.get("gateway", {}).get("telegram", {})
        return bool(tg_cfg.get("topic_mode", False))
    except Exception:
        return False


def is_telegram_topic_root_lobby(source: Any) -> bool:
    """Check if a source is the Telegram topic root lobby (the main group chat)."""
    chat_type = getattr(source, "chat_type", "")
    thread_id = getattr(source, "thread_id", None)
    return chat_type == "group" and thread_id is None


def is_telegram_topic_lane(source: Any) -> bool:
    """Check if a source is a Telegram topic lane (a specific topic thread)."""
    chat_type = getattr(source, "chat_type", "")
    thread_id = getattr(source, "thread_id", None)
    return chat_type == "group" and thread_id is not None


def should_send_telegram_lobby_reminder(config: dict, source: Any) -> bool:
    """Check if we should send a lobby reminder message about topics."""
    try:
        tg_cfg = config.get("gateway", {}).get("telegram", {})
        return bool(tg_cfg.get("lobby_reminder", True))
    except Exception:
        return True


def telegram_topic_root_lobby_message(config: dict) -> str:
    """Get the lobby message shown in the root topic."""
    try:
        tg_cfg = config.get("gateway", {}).get("telegram", {})
        msg = tg_cfg.get("lobby_message", "")
        if msg:
            return str(msg)
    except Exception:
        pass
    return (
        "Welcome! This is the main chat. Send a message here to start, "
        "or use topic threads for specific conversations."
    )


def telegram_topic_root_new_message(config: dict) -> str:
    """Get the new message shown in the root topic for new conversations."""
    try:
        tg_cfg = config.get("gateway", {}).get("telegram", {})
        msg = tg_cfg.get("root_new_message", "")
        if msg:
            return str(msg)
    except Exception:
        pass
    return "New conversation started. What would you like help with?"


def telegram_topic_new_header(source: Any) -> Optional[str]:
    """Get a topic header for display when a new topic thread starts."""
    thread_title = getattr(source, "thread_title", None) or getattr(source, "topic_name", None)
    if thread_title:
        return f"📂 Topic: {thread_title}"
    return None


def record_telegram_topic_binding(
    bindings: Dict[str, str],
    session_id: str,
    thread_id: str,
) -> None:
    """Record a session_id → thread_id binding for topic recovery."""
    bindings[session_id] = thread_id


def recover_telegram_topic_thread_id(
    bindings: Dict[str, str],
    session_id: str,
) -> Optional[str]:
    """Recover thread_id from session_id binding."""
    return bindings.get(session_id)
