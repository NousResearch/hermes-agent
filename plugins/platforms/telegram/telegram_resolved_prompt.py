"""Resolved-prompt edits that keep the request visible, kept out of the adapter facade.

Replacing an approval or slash-confirm prompt with only the decision ("✅ Approved once by Ana")
erases what was approved, so a shared chat loses its record. The edit keeps the prompt's own
rendering (``Message.text_html`` rebuilds it from the entities, code block included), drops the
"if you don't answer within N minutes" line that no longer applies, and appends the decision.
"""

from __future__ import annotations

import html
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Telegram's wording when ``answerCallbackQuery`` arrives after the query's window closed.
_EXPIRED_QUERY_PHRASES = ("query is too old", "query id is invalid", "response timeout expired")


async def answer_refusal(query: Any, text: str) -> None:
    """Show the requester-refusal alert. A tap on a card old enough that Telegram refuses the
    answer is dropped quietly: the approval stays pending for its requester either way, and a
    raise would only fail the update in PTB. Any other answer failure still raises."""
    try:
        await query.answer(text=text, show_alert=True)
    except Exception as exc:
        if not any(phrase in str(exc).lower() for phrase in _EXPIRED_QUERY_PHRASES):
            raise
        try:
            from telegram.error import BadRequest
        except ImportError:
            BadRequest = None
        if BadRequest is not None and not isinstance(exc, BadRequest):
            raise
        logger.debug("Telegram refusal toast dropped: the callback query expired", exc_info=True)


async def edit_keeping_prompt(adapter: Any, query: Any, decision: str) -> None:
    """Append *decision* to the prompt and drop its buttons; the adapter's decision-only edit runs when
    the prompt has no readable body or the appended line pushes it past Telegram's length cap."""
    from telegram.error import BadRequest, NetworkError

    body = getattr(query.message, "text_html", None) if query.message else None
    if isinstance(body, str) and body:
        body = body.replace(adapter._ea_deadline_line(), "")  # same escaping as text_html
        try:
            await query.edit_message_text(text=f"{body}\n\n— {html.escape(decision)}", parse_mode="HTML", reply_markup=None)
            return
        except BadRequest:  # too long / unparsable: the decision-only edit below still clears the buttons
            pass
        except NetworkError as exc:  # network/timeout: the edit may have landed, so keep the prompt as is
            logger.debug("Telegram resolved-prompt edit failed: %s", exc)
            return
    await adapter._edit_md_quiet(query, decision)
