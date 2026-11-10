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


def adopt_routing(event: MessageEvent, routed: MessageEvent) -> None:
    """Give ``event`` the session routing already resolved on ``routed``: its source (identity kept)
    and the topic's prompt inputs, marked so a later merge keeps it. Every site that folds a
    question's turn into another event goes through here, so the merged turn never re-derives its
    topic from the newest binding at dispatch (a question recovered into topic 10 must not move)."""
    event.source = routed.source
    event.auto_skill, event.channel_prompt = routed.auto_skill, routed.channel_prompt
    event._routing_pinned = True  # type: ignore[attr-defined]


def merge_batched_event(pending: dict, key: str, event: MessageEvent, *, merge_caption, route_key) -> str:
    """Retain each attachment's inlining semantics when combining album items; return the lane used.

    Two events whose routing is already resolved (pinned) to different sessions never merge: a second
    question+photo pair recovered into topic 20 must not run in the first pair's topic 10 just because
    both landed in one burst window. The later one takes its own lane keyed by its resolved session.
    """
    existing = pending.get(key)
    if (existing is not None and getattr(event, "_routing_pinned", False) and getattr(existing, "_routing_pinned", False)
            and route_key(existing) != route_key(event)):
        key = f"{key}|{route_key(event)}"
        existing = pending.get(key)
    if existing is None:
        pending[key] = event
        return key
    if getattr(event, "_routing_pinned", False) and not getattr(existing, "_routing_pinned", False):
        adopt_routing(existing, event)  # a later album item absorbed the question: the album joins its topic
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
    return key
