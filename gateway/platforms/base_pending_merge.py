"""Merge operations for pending gateway events."""

from typing import Dict, Optional

from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base_pending import merge_recorded


def _append_text(existing: Optional[str], new: Optional[str]) -> str:
    """``existing\\nnew`` when both non-empty; the non-empty one otherwise."""
    return f"{existing}\n{new}" if existing else new


def _absorb_pending_media(existing: MessageEvent, event: MessageEvent) -> None:
    from gateway.platforms.base import BasePlatformAdapter

    existing.absorb_context_dependencies(event)
    existing_type = existing.message_type
    existing_is_photo = existing_type == MessageType.PHOTO
    incoming_is_photo = event.message_type == MessageType.PHOTO
    both_photo = existing_is_photo and incoming_is_photo
    incoming_has_media = bool(event.media_urls)
    if both_photo or incoming_has_media:
        existing.absorb_media(event)
    if event.text:
        existing.text = BasePlatformAdapter._merge_caption(existing.text, event.text)
    existing.absorb_message_ids(event)
    existing.absorb_reply_context(event)
    existing.absorb_reply_expected(event)
    if existing_is_photo or incoming_is_photo:
        existing.message_type = MessageType.PHOTO
    elif existing_type == MessageType.TEXT and event.message_type != MessageType.TEXT:
        existing.message_type = event.message_type
    for attr in ("_gateway_pending_stt_text", "_gateway_pending_stt_transcripts", "_gateway_pending_stt_clips"):
        if hasattr(existing, attr):
            delattr(existing, attr)


def _absorb_pending_text(existing: MessageEvent, event: MessageEvent) -> None:
    existing.absorb_context_dependencies(event)
    if event.text:
        existing.text = _append_text(existing.text, event.text)
    existing.absorb_message_ids(event)
    existing.absorb_reply_context(event)
    existing.absorb_reply_expected(event)


def _append_batched_text(existing: MessageEvent, event: MessageEvent) -> None:
    existing.absorb_context_dependencies(event)
    if event.text:
        existing.text = _append_text(existing.text, event.text)
    if event.media_urls:
        existing.absorb_media(event)
    existing.absorb_message_ids(event)
    existing.absorb_reply_context(event)
    existing.absorb_reply_expected(event)


def _append_debounced_text(existing: MessageEvent, event: MessageEvent) -> None:
    existing.absorb_context_dependencies(event)
    if event.text:
        existing.text = _append_text(existing.text, event.text)
    if event.media_urls:
        existing.media_text_inlined.extend(
            [None] * (len(existing.media_urls) - len(existing.media_text_inlined))
        )
        existing.media_urls.extend(event.media_urls)
        existing.media_types.extend(event.media_types)
        existing.media_text_inlined.extend(event.media_text_inlined)
        existing.media_text_inlined.extend(
            [None] * (len(existing.media_urls) - len(existing.media_text_inlined))
        )
    existing.absorb_reply_context(event)
    existing.absorb_reply_expected(event)
    latest_message_id = getattr(event, "message_id", None)
    if latest_message_id is not None:
        existing.merged_message_ids.extend(
            message_id for message_id in (existing.message_id, *event.merged_message_ids)
            if message_id
        )
        existing.message_id = str(latest_message_id)
    else:
        existing.absorb_message_ids(event)


def merge_pending_message_event(pending_messages: dict[str, MessageEvent], session_key: str,
                                event: MessageEvent, *, merge_text: bool = False) -> None:
    """Store or merge a pending event: photo bursts/albums merge into the queued event so the next
    turn sees the whole burst; with ``merge_text`` rapid TEXT follow-ups append instead of
    replace."""
    existing = pending_messages.get(session_key)
    merge = None
    if existing:
        existing_type = getattr(existing, "message_type", None)
        # A photo burst always absorbs; otherwise merge only when media is involved on either
        # side. Captions merge in every absorbing case.
        if ((existing_type == MessageType.PHOTO and event.message_type == MessageType.PHOTO)
                or existing.media_urls or event.media_urls):
            merge = _absorb_pending_media
        elif merge_text and existing_type == MessageType.TEXT and event.message_type == MessageType.TEXT:
            merge = _absorb_pending_text
    if existing is None or merge is None:
        pending_messages[session_key] = event
        return
    merge_recorded(existing, event, merge)
