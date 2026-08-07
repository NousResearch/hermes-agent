"""Origin-session binding for Kanban task completion delivery.

Every task-creation surface must use this module after writing the task.  It
turns the current human conversation's session context into a durable
``kanban_notify_subs`` row without treating unattached CLI/cron processes as
message destinations.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _session_env(name: str, default: str = "") -> str:
    """Read request-local routing context with a CLI environment fallback."""
    try:
        from gateway.session_context import get_session_env

        return get_session_env(name, default)
    except Exception:
        return os.environ.get(name, default)


def current_session_id() -> Optional[str]:
    """Return the current durable conversation id when one is bound."""
    return (_session_env("HERMES_SESSION_ID", "") or "").strip() or None


def maybe_auto_subscribe(
    conn: Any,
    task_id: str,
    *,
    session_id: Optional[str] = None,
) -> bool:
    """Subscribe the originating human conversation to terminal task events.

    Messaging platforms bind an explicit platform/chat/thread envelope.  The
    Desktop/TUI uses a local poller keyed by ``HERMES_SESSION_KEY``.  The
    durable ``HERMES_SESSION_ID`` is a different identity and must never be
    substituted for that live poller address.

    Best effort: task creation must still succeed if notification bookkeeping
    fails.  Returns whether a subscription row was written.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        if not cfg_get(cfg, "kanban", "auto_subscribe_on_create", default=True):
            return False
    except Exception:
        # Keep the feature's documented default-on behavior when configuration
        # is temporarily unreadable.
        pass

    platform = ""
    chat_id = ""
    try:
        platform = _session_env("HERMES_SESSION_PLATFORM", "").strip()
        chat_id = _session_env("HERMES_SESSION_CHAT_ID", "").strip()
        if not platform or not chat_id:
            session_key = (
                _session_env("HERMES_SESSION_KEY", "")
                or os.environ.get("HERMES_SESSION_KEY", "")
            ).strip()
            if not session_key:
                return False
            platform = "tui"
            chat_id = session_key

        thread_id = _session_env("HERMES_SESSION_THREAD_ID", "").strip() or None
        user_id = _session_env("HERMES_SESSION_USER_ID", "").strip() or None
        chat_type = _session_env("HERMES_SESSION_CHAT_TYPE", "").strip() or None
        message_id = _session_env("HERMES_SESSION_MESSAGE_ID", "").strip()
        notifier_profile = (
            _session_env("HERMES_SESSION_PROFILE", "").strip()
            or os.environ.get("HERMES_PROFILE", "").strip()
        )
        if not notifier_profile:
            try:
                from hermes_cli.profiles import get_active_profile_name

                notifier_profile = get_active_profile_name() or "default"
            except Exception:
                notifier_profile = "default"

        delivery_metadata: dict[str, Any] = {}
        if thread_id:
            delivery_metadata["thread_id"] = thread_id
        if chat_type:
            delivery_metadata["chat_type"] = chat_type
        if (
            platform.lower() == "telegram"
            and thread_id
            and (chat_type or "").lower() in {"dm", "direct", "private"}
        ):
            delivery_metadata["telegram_dm_topic_reply_fallback"] = True
            if str(thread_id) not in {"", "1"}:
                delivery_metadata["direct_messages_topic_id"] = str(thread_id)
            if message_id:
                delivery_metadata["telegram_reply_to_message_id"] = message_id

        from hermes_cli import kanban_db as kb

        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=thread_id,
            user_id=user_id,
            notifier_profile=notifier_profile,
            delivery_metadata=delivery_metadata or None,
        )
        return True
    except Exception as exc:
        logger.warning(
            "kanban origin auto-subscribe failed: %r (platform=%r target_set=%r)",
            exc,
            platform,
            bool(chat_id),
        )
        return False
