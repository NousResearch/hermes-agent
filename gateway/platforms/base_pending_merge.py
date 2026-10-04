"""Merge operations for pending gateway events."""

from typing import Optional, overload

from gateway.platforms.event import MessageEvent, MessageType


@overload
def _append_text(existing: Optional[str], new: str) -> str: ...


@overload
def _append_text(existing: Optional[str], new: Optional[str]) -> Optional[str]: ...


def _append_text(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    """``existing\\nnew`` when both non-empty; the non-empty one otherwise."""
    return f"{existing}\n{new}" if existing else new


def merge_pending_message_event(
    pending_messages: dict[str, MessageEvent],
    session_key: str,
    event: MessageEvent,
    *,
    merge_text: bool = False,
) -> None:
    """Store or merge a pending event: photo bursts/albums merge into the queued event so the next
    turn sees the whole burst; with ``merge_text`` rapid TEXT follow-ups append instead of
    replace."""
    existing = pending_messages.get(session_key)
    if existing:
        existing_type = getattr(existing, "message_type", None)
        existing_is_photo = existing_type == MessageType.PHOTO
        incoming_is_photo = event.message_type == MessageType.PHOTO
        both_photo = existing_is_photo and incoming_is_photo
        incoming_has_media = bool(event.media_urls)

        # A photo burst always absorbs; otherwise merge only when media is involved on either
        # side. Captions merge in every absorbing case.
        if both_photo or existing.media_urls or incoming_has_media:
            existing.absorb_context_dependencies(event)
            if both_photo or incoming_has_media:
                existing.absorb_media(event)
            if event.text:
                from gateway.platforms.base import BasePlatformAdapter

                existing.text = BasePlatformAdapter._merge_caption(
                    existing.text, event.text
                )
            existing.absorb_message_ids(event)
            existing.absorb_reply_context(event)
            existing.absorb_reply_expected(event)
            if existing_is_photo or incoming_is_photo:
                existing.message_type = MessageType.PHOTO
            elif (
                existing_type == MessageType.TEXT
                and event.message_type != MessageType.TEXT
            ):
                existing.message_type = event.message_type
            # Drop the *derived* STT cache (event changed); the echo ledger must survive or
            # notes echo twice.
            for attr in (
                "_gateway_pending_stt_text",
                "_gateway_pending_stt_transcripts",
            ):
                if hasattr(existing, attr):
                    delattr(existing, attr)
            return
        both_text = (
            existing_type == MessageType.TEXT and event.message_type == MessageType.TEXT
        )
        if merge_text and both_text:
            existing.absorb_context_dependencies(event)
            if event.text:
                existing.text = _append_text(existing.text, event.text)
            existing.absorb_message_ids(event)
            existing.absorb_reply_context(event)
            existing.absorb_reply_expected(event)
            return
    pending_messages[session_key] = event

