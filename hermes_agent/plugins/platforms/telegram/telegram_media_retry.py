"""Transient-failure policy for one native Telegram media upload (#62079).

A media send used to make one attempt: a connect error or Telegram's own 5xx answer went straight
to the "Couldn't deliver the file attachment" notice (or, for a photo, a re-upload as a document).
Now the SAME request is repeated, with the text lane's budget, only when the failure proves nothing
was posted (``telegram_send_errors``: ``UNSENT`` connect/pool failure, ``SERVER_ERROR`` Telegram's
own 5xx JSON). A flood wait keeps its own recovery (``telegram_media_flood``). A failure after the
request left (a read error or timeout after the upload, a 504, an HTML 5xx page) may already be in
the chat: it is never resent in any shape and is reported
``ambiguous`` (``success`` with ``raw_response["ambiguous"]``, the relay's contract), so neither the
failure notice nor a consumer that re-sends on failure (the send_message tool, cron) duplicates it.
Telegram's 5xx JSON on every try is a definite failure: it keeps the notice and the fallback.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from hermes_agent.gateway.platforms.base import SendResult
from hermes_agent.plugins.platforms.telegram import telegram_send_errors as kinds

logger = logging.getLogger("hermes_agent.plugins.platforms.telegram.adapter")

_RETRYABLE = (kinds.UNSENT, kinds.SERVER_ERROR)
# Set on an error the upload request itself raised: only those can mean "it may be in the chat".
# A local error (opening the file, a transcode) classifies the same way by type and never left.
_REQUEST_ERROR_ATTR = "_hermes_telegram_media_request_error"


async def send_media_retrying(
    send_once: Callable[[], Awaitable[Any]], reset_media: Optional[Callable[[], Any]], what: str,
) -> Any:
    """``await send_once()``, repeated up to ``SAME_REQUEST_ATTEMPTS`` times while the failure is
    one Telegram provably never posted; ``reset_media`` rewinds the upload before each repeat."""
    for attempt in range(kinds.SAME_REQUEST_ATTEMPTS):
        try:
            return await send_once()
        except Exception as exc:
            if attempt >= kinds.SAME_REQUEST_ATTEMPTS - 1 or kinds.send_error_kind(exc) not in _RETRYABLE:
                setattr(exc, _REQUEST_ERROR_ATTR, True)
                raise
            from hermes_agent.plugins.platforms.telegram.adapter import _redact_telegram_error_text
            delay = kinds.same_request_backoff(attempt)
            logger.warning("[Telegram] %s not sent (attempt %d/%d), retrying in %.0fs: %s", what, attempt + 1,
                           kinds.SAME_REQUEST_ATTEMPTS, delay, _redact_telegram_error_text(exc))
            await asyncio.sleep(delay)
            if reset_media is not None:
                reset_media()
    raise AssertionError("unreachable")  # pragma: no cover - the last attempt returns or raises


def request_may_have_delivered(error: Exception) -> bool:
    """True when the upload REQUEST raised and may already be in the chat (never resend it): only a
    ``LOST`` request. A ``SERVER_ERROR`` that spent the budget is Telegram's own answer that nothing
    was posted, a definite failure that keeps the notice / fallback."""
    return bool(getattr(error, _REQUEST_ERROR_ATTR, False)) and kinds.send_error_kind(error) == kinds.LOST


def unconfirmed_media_result(error: Exception, what: str, *, as_failure: bool = False) -> Optional[SendResult]:
    """The ambiguous result for an upload request that may already be in the chat, else ``None``.
    ``as_failure``: the URL photo/animation senders keep their failed result (no consumer resends
    a URL send), marked uncertain and notice-suppressed instead of falling back to a re-upload."""
    if not request_may_have_delivered(error):
        return None
    from hermes_agent.plugins.platforms.telegram.adapter import _redact_telegram_error_text
    safe_error = _redact_telegram_error_text(error)
    logger.warning("[Telegram] %s unconfirmed (it may already be in the chat, not resending): %s", what, safe_error)
    if as_failure:
        return SendResult(success=False, error=safe_error, retryable=False,
                          raw_response={"delivery_uncertain": True, "failure_notice_sent": True})
    return SendResult(success=True, raw_response={"ambiguous": True, "error": safe_error})
