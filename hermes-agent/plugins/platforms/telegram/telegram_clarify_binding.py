"""Bind a Telegram clarify prompt to the message that carries its buttons.

A ``cl:<id>:<idx>`` callback is accepted only from the exact ``(chat_id, message_id)`` the clarify was
rendered on. Chat-level authorization proves the caller may answer *some* prompt in that chat; it
cannot tell a deliberate tap on this prompt from a callback that arrived on another message (#102957).
A mismatch leaves the entry pending so the real prompt stays answerable, and is logged with enough
non-secret provenance (callback-query id, caller id, both message identities) to audit afterwards.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _ident(value: Any) -> Optional[str]:
    """Normalize a Telegram id (int on the wire, str in state) for comparison."""
    return None if value is None else str(value)


@dataclasses.dataclass(frozen=True)
class ClarifyPrompt:
    """A pending clarify's session key plus the identity of the message that rendered its buttons.
    ``thread_id`` is for the audit line only: the message implies it, and Telegram omits it for a
    forum's General topic, so gating on it would only add false rejects."""

    session_key: str
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    thread_id: Optional[str] = None

    @classmethod
    def from_sent(cls, session_key: str, msg: Any) -> ClarifyPrompt:
        return cls(session_key, _ident(getattr(msg, "chat_id", None)), _ident(getattr(msg, "message_id", None)),
                   _ident(getattr(msg, "message_thread_id", None)))


async def tap_matches_prompt(query: Any, clarify_id: str, prompt: ClarifyPrompt, denial: str) -> bool:
    """True when the tap came from the prompt's own message; otherwise answer ``denial`` and fail closed."""
    if prompt.message_id is None:
        # Only a Bot API send result can leave no message id (no caller input reaches this): resolve
        # rather than strand the prompt.
        logger.warning("Telegram clarify callback unverifiable: no prompt message bound (id=%s)", clarify_id)
        return True
    message = getattr(query, "message", None)
    got_chat, got_msg = _ident(getattr(message, "chat_id", None)), _ident(getattr(message, "message_id", None))
    if (got_chat, got_msg) == (prompt.chat_id, prompt.message_id):
        return True
    logger.warning(
        "Telegram clarify callback rejected: prompt binding mismatch (id=%s, expected chat=%s message=%s thread=%s, "
        "got chat=%s message=%s thread=%s, callback_query=%s, caller_id=%s)", clarify_id, prompt.chat_id,
        prompt.message_id, prompt.thread_id, got_chat, got_msg, _ident(getattr(message, "message_thread_id", None)),
        getattr(query, "id", None), _ident(getattr(getattr(query, "from_user", None), "id", None)))
    with contextlib.suppress(Exception):
        await query.answer(text=denial)
    return False


def log_resolved(query: Any, clarify_id: str, prompt: ClarifyPrompt, choice: str, user_display: str) -> None:
    """Audit line for a resolved clarify: which caller, which callback query, which prompt message."""
    logger.info(
        "Telegram clarify button resolved (id=%s, choice_len=%d, caller_id=%s, callback_query=%s, chat=%s, message=%s)",
        clarify_id, len(choice), _ident(getattr(getattr(query, "from_user", None), "id", None)),
        getattr(query, "id", None), prompt.chat_id, prompt.message_id)
    logger.debug("Telegram clarify choice text (id=%s, user=%s): %.100r", clarify_id, user_display, choice)
