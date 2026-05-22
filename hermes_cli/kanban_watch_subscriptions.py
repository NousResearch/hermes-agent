from __future__ import annotations

import os
from typing import Any, Optional


SESSION_EVENT_DELIVERY_MODE = "session_event"


def _get_session_env(name: str) -> str:
    try:
        from gateway.session_context import get_session_env
    except Exception:
        get_session_env = None
    value = ""
    if get_session_env is not None:
        try:
            value = get_session_env(name, "")
        except Exception:
            value = ""
    if not value:
        value = os.getenv(name, "")
    return str(value or "").strip()


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def current_session_watch_subscription() -> Optional[dict[str, Optional[str]]]:
    session_key = _get_session_env("HERMES_SESSION_KEY")
    platform = _get_session_env("HERMES_SESSION_PLATFORM").lower()
    chat_id = _get_session_env("HERMES_SESSION_CHAT_ID")
    thread_id = _get_session_env("HERMES_SESSION_THREAD_ID")
    user_id = _get_session_env("HERMES_SESSION_USER_ID")
    if not session_key or not platform or not chat_id:
        return None
    return {
        "delivery_mode": SESSION_EVENT_DELIVERY_MODE,
        "session_key": session_key,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id or None,
        "user_id": user_id or None,
        "notifier_profile": _active_profile_name(),
    }


def resolve_create_task_watch_subscriptions(kb: Any, conn: Any) -> list[dict[str, Optional[str]]]:
    current_task_id = str(os.getenv("HERMES_KANBAN_TASK", "") or "").strip()
    if current_task_id:
        inherited: list[dict[str, Optional[str]]] = []
        for sub in kb.list_notify_subs(conn, current_task_id):
            if (sub.get("delivery_mode") or "message") != SESSION_EVENT_DELIVERY_MODE:
                continue
            if not sub.get("session_key") or not sub.get("platform") or not sub.get("chat_id"):
                continue
            inherited.append({
                "delivery_mode": SESSION_EVENT_DELIVERY_MODE,
                "session_key": str(sub.get("session_key") or ""),
                "platform": str(sub.get("platform") or "").lower(),
                "chat_id": str(sub.get("chat_id") or ""),
                "thread_id": str(sub.get("thread_id") or "") or None,
                "user_id": (str(sub.get("user_id") or "") or None),
                "notifier_profile": (str(sub.get("notifier_profile") or "") or None),
            })
        if inherited:
            return inherited
    current = current_session_watch_subscription()
    return [current] if current else []
