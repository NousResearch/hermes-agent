"""Media predicates and the media-only placeholder for inbound gateway events.

Split out of ``gateway/run.py`` to keep that facade inside its code-health line
cap; the names stay importable from ``gateway.run``.
"""

from gateway.platforms.event import MessageType


def _event_media_type_at(event, index: int) -> str:
    """Per-attachment MIME at *index*; "" when the adapter set only a message-level type."""
    media_types = getattr(event, "media_types", None) or []
    return media_types[index] if index < len(media_types) else ""


def _event_media_kind_is(event, index: int, mime_prefix: str, fallback_types: frozenset) -> bool:
    """Per-attachment MIME first, message-level type only when unknown (else a document uploaded
    alongside an image is base64'd as vision and the provider 400s)."""
    mtype = _event_media_type_at(event, index)
    if mtype:
        return mtype.startswith(mime_prefix)
    return getattr(event, "message_type", None) in fallback_types


def _event_media_is_image(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "image/", frozenset({MessageType.PHOTO}))


def _event_media_is_audio(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "audio/", frozenset({MessageType.VOICE, MessageType.AUDIO}))


def _event_media_is_stt_input(event, index: int) -> bool:
    """True when an audio attachment should enter the automatic STT pipeline."""
    message_type = getattr(event, "message_type", None)
    if message_type in {MessageType.AUDIO, MessageType.DOCUMENT}:
        return False
    # Telegram round video notes arrive as video attachments that carry an audio track:
    # the adapter marks them, and that audio is transcription input like a voice note.
    # Plain videos stay vision-only.
    if (
        (getattr(event, "metadata", None) or {}).get("telegram_video_note")
        and _event_media_type_at(event, index).startswith("video/")
    ):
        return True
    return message_type == MessageType.VOICE or _event_media_type_at(event, index).startswith("audio/")


def _event_media_is_video(event, index: int) -> bool:
    return _event_media_kind_is(event, index, "video/", frozenset({MessageType.VIDEO}))


def _build_media_placeholder(event) -> str:
    """Text placeholder for media-only events (later replaced by vision enrichment).
    Queued media is dequeued via .text only, so a caption-less event would otherwise be lost."""
    parts = []
    media_urls = getattr(event, "media_urls", None) or []
    for i, url in enumerate(media_urls):
        if _event_media_is_image(event, i):
            parts.append(f"[User sent an image: {url}]")
        elif _event_media_is_audio(event, i):
            parts.append(f"[User sent audio: {url}]")
        elif _event_media_is_video(event, i):
            parts.append(f"[User sent a video: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)
