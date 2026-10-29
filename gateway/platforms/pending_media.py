"""Attachment identity metadata retained while inbound media events merge."""
from typing import Optional

from gateway.platforms.event import MessageEvent, MessageType


def _merge_pending_media_metadata(existing: MessageEvent, incoming: MessageEvent) -> None:
    """Merge only attachment identity metadata that remains valid after media concatenation."""
    incoming_metadata = incoming.metadata if isinstance(incoming.metadata, dict) else {}
    values = incoming_metadata.get("discord_path_only_attachment_paths")
    if not isinstance(values, list) or not values:
        return
    if not isinstance(existing.metadata, dict):
        existing.metadata = {}
    current = existing.metadata.setdefault("discord_path_only_attachment_paths", [])
    if not isinstance(current, list):
        current = []
        existing.metadata["discord_path_only_attachment_paths"] = current
    for value in values:
        if value not in current:
            current.append(value)


def merge_pending_message_event(pending_messages: dict[str, MessageEvent], session_key: str,
                                event: MessageEvent, *, merge_text: bool = False) -> None:
    """Store or merge a pending event: photo bursts/albums merge into the queued event so the next
    turn sees the whole burst; with ``merge_text`` rapid TEXT follow-ups append instead of
    replace."""
    from gateway.platforms.base import BasePlatformAdapter, _append_text
    existing = pending_messages.get(session_key)
    if existing:
        existing_type = getattr(existing, "message_type", None)
        existing_is_photo = existing_type == MessageType.PHOTO
        incoming_is_photo = event.message_type == MessageType.PHOTO
        both_photo = existing_is_photo and incoming_is_photo
        incoming_has_media = bool(event.media_urls)

        def _padded_inline_flags(msg: MessageEvent) -> list[Optional[bool]]:
            flags = list(getattr(msg, "media_text_inlined", []) or [])
            return flags + [None] * (len(msg.media_urls) - len(flags))
        incoming_inline_flags: list[Optional[bool]] = []
        if incoming_has_media:
            existing.media_text_inlined = _padded_inline_flags(existing)
            incoming_inline_flags = _padded_inline_flags(event)
        # A photo burst always absorbs; otherwise merge only when media is involved on either
        # side. Captions merge in every absorbing case.
        if both_photo or existing.media_urls or incoming_has_media:
            if both_photo or incoming_has_media:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
                existing.media_text_inlined.extend(incoming_inline_flags)
                _merge_pending_media_metadata(existing, event)
            if event.text:
                existing.text = BasePlatformAdapter._merge_caption(existing.text, event.text)
            existing.absorb_reply_expected(event)
            if existing_is_photo or incoming_is_photo:
                existing.message_type = MessageType.PHOTO
            elif existing_type == MessageType.TEXT and event.message_type != MessageType.TEXT:
                existing.message_type = event.message_type
            # Drop the *derived* STT cache (event changed); the echo ledger must survive or
            # notes echo twice.
            for attr in ("_gateway_pending_stt_text", "_gateway_pending_stt_transcripts"):
                if hasattr(existing, attr):
                    delattr(existing, attr)
            return
        both_text = existing_type == MessageType.TEXT and event.message_type == MessageType.TEXT
        if merge_text and both_text:
            if event.text:
                existing.text = _append_text(existing.text, event.text)
            existing.absorb_reply_expected(event)
            return
    pending_messages[session_key] = event
