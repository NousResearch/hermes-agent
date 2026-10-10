"""Shutdown/restart logging and notice dedup keys, split out of ``gateway.run_shutdown``
(that module is past its FILE_LINES cap; moved code keeps its cap — see AGENTS.md).
``gateway.run_shutdown`` re-imports all three names, so ``from gateway.run_shutdown import
_log_suppressed`` and module-level monkeypatching keep working.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


@contextmanager
def _log_suppressed(level: int, msg: str, *args, exc_info: bool = False):
    """``suppress(Exception)`` that logs the swallowed exception on ``gateway.run``.

    Without ``exc_info`` the exception is appended as the last ``%s`` argument (``msg % (*args, exc)``);
    with it the traceback is attached instead. Best-effort seams use this everywhere a failure must be
    visible in the log but must never propagate.
    """
    try:
        yield
    except Exception as exc:
        if exc_info:
            logger.log(level, msg, *args, exc_info=(type(exc), exc, exc.__traceback__))
        else:
            logger.log(level, msg, *args, exc)


def _notice_target_key(platform_value: str, chat_id, thread_id) -> tuple:
    """Dedup key for one notice destination: thread/topic platforms share a chat but route apart."""
    return (platform_value, str(chat_id), str(thread_id) if thread_id else None)


def _delivery_target_key(platform_value: str, chat_id, thread_id, *, profile: Optional[str] = None) -> tuple:
    """Dedupe key for one DELIVERED chat: profile-independent, except Telegram private chats.

    Two served profiles can share one home chat (one Telegram group for the whole host) and owe it
    ONE notice per host restart. A positive Telegram chat id names the USER, though: the same id
    under two bot tokens is two conversations, so those stay keyed per served profile (#118233).
    """
    from gateway.delivery import looks_like_telegram_private_chat_id
    if (profile and profile != "default" and platform_value == "telegram"
            and looks_like_telegram_private_chat_id(chat_id)):
        platform_value = f"{profile}:{platform_value}"
    return _notice_target_key(platform_value, chat_id, thread_id)
