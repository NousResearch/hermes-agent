"""Finishing a streamed preview in place when the turn's final was not confirmed delivered.

A Telegram flood refusal (or a cancelled stream consumer) can leave the preview frozen mid-answer
with every delivery flag unset. A plain final send then posts the whole answer again next to it.
The preview is the answer's own message, so the gateway edits it up to the complete text instead
(an edit to identical text is a no-op), waiting out flood refusals the way the final send would
(each wait at most ``_SEND_RETRY_INLINE_WAIT_CAP_SECS``) instead of failing closed into a second copy.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Optional

from hermes_agent.gateway.delivery_ledger import flood_wait_seconds, is_flood_error
from hermes_agent.gateway.platforms.base import _SEND_RETRY_INLINE_WAIT_CAP_SECS

logger = logging.getLogger("hermes_agent.gateway.run")

_FLOOD_WAIT_SLACK_SECS = 1.0
_MAX_FLOOD_WAITS = 2

# ``edit_preview_to_final`` result when the turn was replaced (/new, /stop) before or during a wait:
# nothing more is owed. Distinct from an adapter edit that returned no receipt (``None``), which the
# callers keep treating as they did before this helper existed.
TURN_REPLACED = object()


def _flood_wait(result: Any) -> Optional[float]:
    """Seconds a flood-refused edit asked to wait, or ``None`` for any other failure."""
    if not is_flood_error(getattr(result, "error", None)):
        return None
    retry_after = getattr(result, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return float(retry_after)
    return flood_wait_seconds(getattr(result, "error", None))


def _edit_carries_whole(adapter: Any, chat_id: str, content: str) -> bool:
    """Whether one ``edit_message(finalize=True)`` can show all of *content*: the adapter splits an
    oversized final edit into continuations itself (``SPLITS_OVERSIZED_FINAL_EDIT``, Telegram), or
    *content* fits one message in the adapter's own length units. Google Chat, Slack and DingTalk
    cut an oversized edit short and still report success, so only the normal send is whole there.
    Slack cuts the FORMATTED text (``&`` -> ``&amp;``), others the raw text: both must fit."""
    if getattr(adapter, "SPLITS_OVERSIZED_FINAL_EDIT", False) is True:
        return True
    from hermes_agent.gateway.platforms.base import BasePlatformAdapter
    limit, len_fn = getattr(adapter, "MAX_MESSAGE_LENGTH", 4096), len
    if isinstance(adapter, BasePlatformAdapter):
        limit, len_fn = adapter.max_message_length_for_chat(chat_id), adapter.message_len_fn_for_chat(chat_id)
    fmt = getattr(adapter, "format_message", None)
    sent = (content, fmt(content)) if callable(fmt) else (content,)
    return isinstance(limit, int) and all(len_fn(text) <= limit for text in sent)


def owns_reconcilable_preview(consumer: Any, chat_id: str = "", content: Optional[str] = None) -> bool:
    """The consumer still holds ONE real preview of this answer, with nothing that may already
    show the answer elsewhere (a timed-out send that may have landed, a split whose id is only the
    last chunk, a native stream), and one edit of it can carry the whole final *content* (as the
    consumer displays it). Editing that
    preview never adds a message, so it cannot duplicate anything."""
    if consumer is None or getattr(consumer, "_turn_split_delivery", False) is True:
        return False
    message_id = getattr(consumer, "message_id", None)
    if not message_id or message_id == "__no_edit__" or getattr(consumer, "adapter", None) is None:
        return False
    if any(getattr(consumer, flag, False) is True for flag in (
            "_use_native_streaming", "_delivery_ambiguous", "delivery_uncertain", "_preview_uncertain",
            "_egress_declined")):
        return False
    if content is None:
        return True
    clean = getattr(consumer, "_clean_for_display", None)
    return _edit_carries_whole(consumer.adapter, chat_id, clean(content) if callable(clean) else content)


def _accepts_metadata(edit_message: Any) -> bool:
    """Same compatibility rule as ``StreamTransportMixin._edit_message``: pass routing metadata only
    to an ``edit_message`` that names it or takes ``**kwargs``."""
    try:
        params = inspect.signature(edit_message).parameters
    except (TypeError, ValueError):
        return False
    return "metadata" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def edit_preview_to_final(adapter: Any, chat_id: str, message_id: str, content: str,
                                metadata: Any = None, *,
                                still_current: Optional[Callable[[], bool]] = None) -> Any:
    """``edit_message(finalize=True)``, retried after each flood refusal the platform asked to wait
    out for at most the final send's inline cap. A split whose head landed while its tail was refused
    resumes only the certain-undelivered tail, never the head again; a tail that still fails is left
    to the caller's normal send (a repeated head beats a lost tail). Routing ``metadata`` rides on
    every call (a split's continuations stay in the topic). ``still_current`` is re-checked before
    the first edit and after every wait: a turn ``/new`` or ``/stop`` replaced meanwhile publishes
    nothing more and :data:`TURN_REPLACED` is returned."""
    def _current() -> bool:
        return still_current is None or bool(still_current())

    kwargs: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "content": content, "finalize": True}
    if metadata and _accepts_metadata(adapter.edit_message):
        kwargs["metadata"] = metadata
    if not _current():
        return TURN_REPLACED
    result = await adapter.edit_message(**kwargs)
    for _ in range(_MAX_FLOOD_WAITS):
        if getattr(result, "success", True):
            return result
        wait = _flood_wait(result)
        if wait is None or wait > _SEND_RETRY_INLINE_WAIT_CAP_SECS:
            return result
        logger.info("Streamed preview %s: final edit flood-refused, retrying in %.0fs", message_id, wait)
        await asyncio.sleep(wait + _FLOOD_WAIT_SLACK_SECS)
        if not _current():
            logger.info("Streamed preview %s: turn no longer current after the flood wait; not finishing it", message_id)
            return TURN_REPLACED
        raw = getattr(result, "raw_response", None)
        if isinstance(raw, dict) and raw.get("partial_overflow"):
            resumed = await adapter._resume_partial_send(chat_id, result, reply_to=None, metadata=metadata)
            if resumed is None:
                return result
            result = resumed
        else:
            result = await adapter.edit_message(**kwargs)
    return result


async def edit_streamed_message(consumer: Any, source: Any, response: dict, content: str, *, sk: str,
                                ok: tuple, fail_result: str, fail_exc: str) -> None:
    """Edit the consumer's message in place with ``content``; on success mark
    ``response["already_sent"]`` and log ``ok``. A returned failure logs ``fail_result`` as
    ``(session, error)`` and an exception logs ``fail_exc`` as ``(session, exc)``; either way
    ``already_sent`` stays unset so the normal final send delivers the content."""
    try:
        result = await edit_preview_to_final(
            consumer.adapter, source.chat_id, consumer.message_id, content, getattr(consumer, "metadata", None),
            still_current=getattr(consumer, "_run_still_current", None))
    except Exception as edit_err:  # health: allow BLE001 -- any edit failure falls back to the normal final send
        logger.warning(fail_exc, sk, edit_err)
        return
    if result is TURN_REPLACED:
        return  # its stale-run discard owns what happens next
    if not getattr(result, "success", True):
        logger.warning(fail_result, sk, getattr(result, "error", None))
        return
    response["already_sent"] = True
    logger.info(*ok)
