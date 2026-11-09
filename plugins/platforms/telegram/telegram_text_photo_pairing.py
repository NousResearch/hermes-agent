"""Telegram text-then-photo pairing: one turn for a question and the image sent right after it.

A user often types a short question and sends the photo it is about a moment later. Telegram
delivers two updates. The text batch flushes after its quiet period (0.18 s for short text), while
the photo is still downloading and then waits in its own 0.8 s burst batch, so the agent answered
the question without the image and then got the image as a second turn (#33270).

Updates of one chat run one after another, so when the image's handler starts, the text it belongs
with is still pending. The handler *claims* that text batch before it awaits anything. Once its normal
quiet period is up, a claimed text flush waits for the image handler (bounded by
``CLAIM_MAX_WAIT_SECONDS``), and the image *absorbs* the text when it reaches the photo batcher.
A lone text, or an image that arrives after the text already flushed, is unchanged.
"""

import asyncio
import contextlib
import copy
import functools
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.telegram_media_group import tracking_album_items

# Upper bound on how long a due text flush waits for a claiming image's download.
CLAIM_MAX_WAIT_SECONDS = 8.0


def _claims(adapter: Any) -> dict[int, tuple[str, MessageEvent, asyncio.Event]]:
    return adapter.__dict__.setdefault("_photo_text_claims", {})


def _is_image(msg: Any) -> bool:
    doc = getattr(msg, "document", None)
    return bool(getattr(msg, "photo", None)) or (
        doc is not None and (getattr(doc, "mime_type", None) or "").lower().startswith("image/"))


def _claim_pending_text(adapter: Any, msg: Any) -> None:
    """Pin a text-only batch from the image's sender in the image's chat that is still waiting to flush."""
    sender = getattr(getattr(msg, "from_user", None), "id", None)
    chat = getattr(getattr(msg, "chat", None), "id", None)
    if not _is_image(msg) or sender is None or chat is None:
        return
    for key, pending in adapter._pending_text_batches.items():
        source = pending.source
        if (source is None or str(source.chat_id) != str(chat) or str(source.user_id) != str(sender)
                or pending.message_type != MessageType.TEXT or pending.media_urls
                or getattr(pending, "_photo_claim", None) is not None):
            continue
        done = pending._photo_claim = asyncio.Event()  # type: ignore[attr-defined]
        _claims(adapter)[id(msg)] = (key, pending, done)
        return


def absorb_claimed_text(adapter: Any, msg: Any, event: MessageEvent) -> None:
    """Move the claimed text into the image ``event`` (text first, then the caption) and cancel its flush.

    Only when the image lands in the same session/topic lane as the text: the key is recomputed on a
    copy because ``_text_batch_key`` applies topic recovery in place.
    """
    claim = _claims(adapter).get(id(msg))
    if claim is None:
        return
    key, pending, _done = claim
    if adapter._pending_text_batches.get(key) is not pending or adapter._text_batch_key(copy.copy(event)) != key:
        return  # flushed already (claim wait expired), or another lane
    adapter._pending_text_batches.pop(key, None)
    task = adapter._pending_text_batch_tasks.pop(key, None)
    if task is not None and not task.done():
        task.cancel()
    event.text = adapter._merge_caption(pending.text, event.text) if event.text else pending.text
    event.absorb_reply_expected(pending)
    if pending.reply_to_message_id and not event.reply_to_message_id:
        for attr in ("reply_to_message_id", "reply_to_text", "reply_to_author_id",
                     "reply_to_author_name", "reply_to_is_own_message"):
            setattr(event, attr, getattr(pending, attr))


def tracking_media_items(handler):
    """Decorate the media handler: album tracking, plus a claim on a pending text that lives until it returns."""
    @functools.wraps(handler)
    async def wrapper(adapter, update, context):
        msg = adapter._effective_update_message(update)
        if msg is not None:
            _claim_pending_text(adapter, msg)  # synchronous: before the handler's first await
        try:
            return await handler(adapter, update, context)
        finally:
            claim = _claims(adapter).pop(id(msg), None) if msg is not None else None
            if claim is not None:
                claim[2].set()
    return tracking_album_items(wrapper)


async def wait_text_flush(adapter: Any, key: str, delay: float) -> None:
    """The text batch's quiet period, then (if an image claimed it) that image's handler, bounded."""
    await asyncio.sleep(delay)
    pending: Optional[MessageEvent] = adapter._pending_text_batches.get(key)
    done = getattr(pending, "_photo_claim", None)
    if done is not None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(done.wait(), CLAIM_MAX_WAIT_SECONDS)
