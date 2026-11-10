"""Telegram inbound album and photo-burst event merging."""

from typing import Optional

from gateway.platforms.event import MessageEvent
from plugins.platforms.telegram.telegram_media_group import item_notes


def pad_inline_flags(event: MessageEvent) -> list[Optional[bool]]:
    """Represent unspecified legacy flags as None before appending media.

    Local twin of core's private ``gateway.platforms.base_media_flags`` helper: plugins may only
    import the published plugin API.
    """
    flags = list(getattr(event, "media_text_inlined", []) or [])
    return flags + [None] * (len(event.media_urls) - len(flags))


def merge_batched_event(pending: dict, key: str, event: MessageEvent, *, merge_caption) -> None:
    """Retain each attachment's inlining semantics when combining album items."""
    existing = pending.get(key)
    if existing is None:
        pending[key] = event
        return
    existing.absorb_reply_expected(event)
    existing.media_text_inlined = pad_inline_flags(existing) + pad_inline_flags(event)
    existing.extend_media(event)
    caption, notes = event.text or "", []
    for note in reversed(item_notes(event)):
        if caption.endswith(note):  # two same-name or unnamed same-size items both keep their note
            caption, notes = caption[:-len(note)].rstrip("\n"), [note, *notes]
    if caption:
        existing.text = merge_caption(existing.text, caption)
    for note in notes:
        existing.text = f"{existing.text}\n\n{note}" if existing.text else note
