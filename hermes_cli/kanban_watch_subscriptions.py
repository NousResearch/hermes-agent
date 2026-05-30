from __future__ import annotations

from typing import Optional


SESSION_EVENT_DELIVERY_MODE = "session_event"


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _parse_session_key(session_key: str) -> Optional[dict[str, str]]:
    """Parse a gateway session key into routing fields.

    Session keys follow ``agent:main:{platform}:{chat_type}:{chat_id}[:extra...]``.
    We only derive ``thread_id`` when the suffix is unambiguous (``dm`` and
    ``thread`` chat types). For group/channel keys the 6th element may be a
    participant id rather than a thread id, so we intentionally leave it unset.
    """
    parts = str(session_key or "").strip().split(":")
    if len(parts) < 5 or parts[0] != "agent" or parts[1] != "main":
        return None
    parsed = {
        "platform": parts[2].strip().lower(),
        "chat_type": parts[3].strip().lower(),
        "chat_id": parts[4].strip(),
    }
    if len(parts) > 5 and parsed["chat_type"] in {"dm", "thread"}:
        parsed["thread_id"] = parts[5].strip()
    return parsed if parsed["platform"] and parsed["chat_id"] else None


def watch_subscription_from_session_key(
    session_key: str,
) -> Optional[dict[str, Optional[str]]]:
    session_key = str(session_key or "").strip()
    if not session_key:
        return None
    parsed = _parse_session_key(session_key)
    if not parsed:
        return None
    return {
        "delivery_mode": SESSION_EVENT_DELIVERY_MODE,
        "session_key": session_key,
        "platform": parsed["platform"],
        "chat_id": parsed["chat_id"],
        "thread_id": parsed.get("thread_id") or None,
        "user_id": None,
        "notifier_profile": _active_profile_name(),
    }


def require_watcher_session_key_subscription(
    session_key: str,
) -> dict[str, Optional[str]]:
    sub = watch_subscription_from_session_key(session_key)
    if sub:
        return sub
    raise ValueError(
        "watcher_session_key must be a valid Hermes gateway session key"
    )
