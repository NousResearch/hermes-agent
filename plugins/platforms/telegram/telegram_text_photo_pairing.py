"""Telegram text-then-photo pairing: one turn for a question and the image sent right after it.

A user often types a short question and sends the photo it is about a moment later. Telegram
delivers two updates. The text batch flushes after its quiet period (0.18 s for short text), while
the photo is still downloading and then waits in its own 0.8 s burst batch, so the agent answered
the question without the image and then got the image as a second turn (#33270).

Updates of one chat run one after another, so when the image's handler starts, the text it belongs
with is still pending. Once the image passed the group gate, the handler *claims* that text batch before it awaits anything. Once its normal
quiet period is up, a claimed text flush waits for the image handler (bounded by
``CLAIM_MAX_WAIT_SECONDS``), and the image *absorbs* the text when it reaches the photo batcher.
A lone text, or an image that arrives after the text already flushed, is unchanged.
"""

import asyncio
import contextlib
import copy
import functools
import os
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.telegram_batching import adopt_routing
from plugins.platforms.telegram.telegram_media_group import tracking_album_items

# Upper bound on how long a due text flush waits for a claiming image's download.
CLAIM_MAX_WAIT_SECONDS = 8.0


def _claims(adapter: Any) -> dict[int, tuple[str, MessageEvent, asyncio.Event]]:
    return adapter.__dict__.setdefault("_photo_text_claims", {})


# Shared with ``TelegramAdapter._cache_inbound_document`` so pairing and the image download path agree.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif", ".avif"})


def is_image_document(doc: Any) -> bool:
    """A document the download path caches as an image: an image filename or an ``image/*`` MIME type."""
    ext = os.path.splitext(getattr(doc, "file_name", None) or "")[1].lower()
    return ext in IMAGE_EXTENSIONS or (getattr(doc, "mime_type", None) or "").lower().startswith("image/")


def _is_image(msg: Any) -> bool:
    doc = getattr(msg, "document", None)
    return bool(getattr(msg, "photo", None)) or (doc is not None and is_image_document(doc))


def claim_pending_text(adapter: Any, msg: Any, event: MessageEvent) -> None:
    """Pin the text-only batch waiting to flush in the lane the image's own ``event`` would batch under.

    Called once the image passed the group gate (an observe-only photo never joins a turn, so it must
    not hold one) and before the handler's first await. The lane is the event's text-batch key (topic
    recovery and observed-group sender included): a photo in topic B never holds topic A's question.
    """
    sender = getattr(getattr(msg, "from_user", None), "id", None)
    if not _is_image(msg) or sender is None:
        return
    key = adapter._text_batch_key(copy.copy(event))  # topic recovery rewrites in place
    pending = adapter._pending_text_batches.get(key)
    if (pending is None or str(pending.sender_id()) != str(sender)  # observed groups keep the sender off the source
            or pending.message_type != MessageType.TEXT or pending.media_urls
            or getattr(pending, "_photo_claim", None) is not None):
        return
    done = pending._photo_claim = asyncio.Event()  # type: ignore[attr-defined]
    _claims(adapter)[id(msg)] = (key, pending, done)


def absorb_claimed_text(adapter: Any, msg: Any, event: MessageEvent) -> None:
    """Move the claimed text into the image ``event`` (text first, then the caption) and cancel its flush.

    Only when the image lands in the same session/topic lane as the text: the key is recomputed on a
    copy because ``_text_batch_key`` applies topic recovery in place. The image then takes that
    lane's routing (the topic the question was already recovered into), so the merged turn cannot
    move to a topic bound later while the photo debounces.
    """
    claim = _claims(adapter).get(id(msg))
    if claim is None:
        return
    key, pending, _done = claim
    routed = copy.copy(event)
    if adapter._pending_text_batches.get(key) is not pending or adapter._text_batch_key(routed) != key:
        return  # flushed already (claim wait expired), or another lane
    adopt_routing(event, routed)
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
    """Decorate the media handler: album tracking, and the release of its text claim when it returns."""
    @functools.wraps(handler)
    async def wrapper(adapter, update, context):
        msg = adapter._effective_update_message(update)
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
