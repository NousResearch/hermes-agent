"""Callback-query answers that survive a tap on an expired Telegram card.

Telegram refuses ``answerCallbackQuery`` once the query is older than its window, with
``BadRequest: Query is too old and response timeout expired or query id is invalid``. The
adapter awaited that call unguarded in every button handler, so a tap on a card that had been
sitting in the chat raised out of the handler: PTB logged the update as failed, whatever the
handler had left to do never ran (a model switch kept its picker state, since the pop sits
after the toast), and the tap produced no feedback at all.

``answer_callback`` is the one choke point for those answers. A refused answer is reported as
``False`` instead of raising. ``answer_or_note`` covers taps on a menu that has already expired —
the one case where the toast was the only thing the user would have seen, and "Picker expired —
use /model again." is still a usable next step as a normal message, posted in the chat and forum
topic the card lives in. Every other error keeps raising: a broken answer must stay visible.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Telegram truncates ``answerCallbackQuery`` text at 200 chars.
TOAST_LIMIT = 200

# Wording of the errors that mean "this tap arrived after the query expired".
_STALE_QUERY_PHRASES = (
    "query is too old",
    "query id is invalid",
    "response timeout expired",
)


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


async def answer_callback(query: Any, text: Optional[str] = None) -> bool:
    """Ack a button tap; ``False`` when Telegram refused it because the query expired."""
    try:
        if text is None:
            await query.answer()
        else:
            await query.answer(text=text)
    except Exception as exc:
        if not is_stale_query_error(exc):
            raise
        logger.debug("[Telegram] callback answer refused for an expired query", exc_info=True)
        return False
    return True


async def answer_or_note(adapter: Any, query: Any, text: str) -> None:
    """Answer an expired-menu tap; when Telegram refused the late answer, post *text* as a normal
    message through the adapter's topic-aware control send, in the tapped card's forum topic."""
    message = getattr(query, "message", None)
    if await answer_callback(query, text) or getattr(message, "chat_id", None) is None:
        return
    thread_id = getattr(message, "message_thread_id", None)
    in_topic = thread_id is not None and getattr(message, "is_topic_message", False) is True
    try:
        await adapter._send_control_message(
            str(message.chat_id), text, parse_mode=None, thread_id=str(thread_id) if in_topic else None, metadata=None)
    except Exception:
        logger.debug("[Telegram] expired-tap chat note could not be sent", exc_info=True)
