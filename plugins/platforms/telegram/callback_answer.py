"""Callback-query answers that survive a tap on an expired Telegram card.

Telegram refuses ``answerCallbackQuery`` once the query is older than its window, with
``BadRequest: Query is too old and response timeout expired or query id is invalid``. The
adapter awaited that call unguarded in every button handler, so a tap on a card that had been
sitting in the chat raised out of the handler: PTB logged the update as failed, whatever the
handler had left to do never ran (a model switch kept its picker state, since the pop sits
after the toast), and the tap produced no feedback at all.

``answer_callback`` is the one choke point for those answers. A refused answer is reported as
``False`` instead of raising, and the caller can ask for a plain chat message when the tap was
for a menu that has already expired — the one case where the toast was the only thing the user
would have seen, and "Picker expired — use /model again." is still a usable next step as a
normal message. Every other error keeps raising: a broken answer must stay visible.
"""

import logging
from typing import Any, Optional

from agent.i18n import t

logger = logging.getLogger(__name__)

# Telegram truncates ``answerCallbackQuery`` text at 200 chars.
TOAST_LIMIT = 200

# Wording of the errors that mean "this tap arrived after the query expired".
_STALE_QUERY_PHRASES = (
    "query is too old",
    "query id is invalid",
    "response timeout expired",
)


def toast(key: str, **kwargs: Any) -> str:
    """``t()`` for ``query.answer(text=...)`` payloads, cut at Telegram's 200-char toast cap."""
    return t(key, **kwargs)[:TOAST_LIMIT]


def is_stale_query_error(exc: BaseException) -> bool:
    """True when *exc* is Telegram refusing an answer because the callback query expired.

    Matched on the wording first (cheap, and it works where python-telegram-bot is not
    importable), then narrowed to PTB's ``BadRequest`` when the library is present so an
    unrelated failure that happens to carry the text still fails loudly.
    """
    text = str(exc).lower()
    if not any(phrase in text for phrase in _STALE_QUERY_PHRASES):
        return False
    try:
        from telegram.error import BadRequest
    except ImportError:
        return True
    return isinstance(exc, BadRequest)


async def answer_callback(
    query: Any, text: Optional[str] = None, *, chat_note: bool = False, chat_id: Optional[str] = None,
) -> bool:
    """Ack a button tap; ``False`` when Telegram refused it because the query expired.

    ``chat_note`` posts *text* into the chat as an ordinary message when the answer could not
    be delivered, so an expired-menu tap still tells the user what to do next.
    """
    try:
        if text is None:
            await query.answer()
        else:
            await query.answer(text=text)
    except Exception as exc:
        if not is_stale_query_error(exc):
            raise
        logger.debug("[Telegram] callback answer refused for an expired query", exc_info=True)
        if chat_note and text:
            await _post_chat_note(query, str(text), chat_id)
        return False
    return True


async def _post_chat_note(query: Any, text: str, chat_id: Optional[str] = None) -> None:
    """Deliver an expired-tap note as a normal message: best effort, never fatal."""
    target = chat_id if chat_id is not None else getattr(getattr(query, "message", None), "chat_id", None)
    if target is None:
        return
    try:
        await query.get_bot().send_message(chat_id=target, text=text)
    except Exception:
        logger.debug("[Telegram] expired-tap chat note could not be sent", exc_info=True)
