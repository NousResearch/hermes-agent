"""Telegram inbound album and photo-burst event merging."""

from gateway.platforms.event import MessageEvent
from gateway.platforms.base_media_flags import pad_inline_flags


def merge_batched_event(pending: dict, key: str, event: MessageEvent, *, merge_caption) -> None:
    """Retain each attachment's inlining semantics when combining album items."""
    existing = pending.get(key)
    if existing is None:
        pending[key] = event
        return
    existing.media_text_inlined = pad_inline_flags(existing) + pad_inline_flags(event)
    existing.extend_media(event)
    if event.text:
        existing.text = merge_caption(existing.text, event.text)
