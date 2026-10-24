"""Telegram inbound album and photo-burst event merging."""

from gateway.platforms.event import MessageEvent


def merge_batched_event(pending: dict, key: str, event: MessageEvent, *, merge_caption) -> None:
    """Retain each attachment's inlining semantics when combining album items."""
    existing = pending.get(key)
    if existing is None:
        pending[key] = event
        return
    # An empty list means legacy semantics, represented by None for each attachment.
    existing.media_text_inlined = (
        (existing.media_text_inlined or [None] * len(existing.media_urls))
        + (event.media_text_inlined or [None] * len(event.media_urls))
    )
    existing.media_urls.extend(event.media_urls)
    existing.media_types.extend(event.media_types)
    if event.text:
        existing.text = merge_caption(existing.text, event.text)
