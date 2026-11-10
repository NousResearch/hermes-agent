"""Telegram inbound album and photo-burst event merging."""

from typing import Any, Optional

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


def resolve_routing(adapter: Any, event: MessageEvent) -> None:
    """Route ``event`` against the topic bindings as they are when it arrives, and pin that route.

    The recovery dispatch applies (``handle_message``), run before the event may join a batch, as the
    text batcher already does: a batch flushes up to seconds later, after another topic may have become
    the newest binding. An internally routed event names its session and is never recovered.
    """
    if event.source is None or getattr(event, "_routing_pinned", False):
        return
    if not str((event.metadata or {}).get("gateway_session_key") or "").strip():
        adapter._apply_topic_recovery(event)
    event._routing_pinned = True  # type: ignore[attr-defined]


def merge_batched_event(pending: dict, key: str, event: MessageEvent, *, merge_caption, route_key) -> str:
    """Retain each attachment's inlining semantics when combining album items; return the lane used.

    Invariant: every event in one batch has the same resolved session key (``route_key``; callers
    route each event with ``resolve_routing`` first, so the key is fixed). An event whose base lane
    holds another route joins (or opens) the ``<key>|<route>`` lane, never the base one: a second
    question+photo pair, and the plain photos after it, stay in topic 20 when topic 10 opened the burst.
    """
    route = route_key(event)
    lane = f"{key}|{route}"
    if lane not in pending and (key not in pending or route_key(pending[key]) == route):
        lane = key
    key, existing = lane, pending.get(lane)
    if existing is None:
        pending[key] = event
        return key
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
