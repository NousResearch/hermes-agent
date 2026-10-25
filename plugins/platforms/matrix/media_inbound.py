"""Inbound Matrix media classification and naming.

Matrix v1.10 media captions: when ``content.filename`` is set and differs from
``body``, ``body`` is the user's caption and the original file name lives in
``filename`` — caching the bytes under the caption loses the extension (#135898).
"""

from gateway.platforms.base import MessageType
from plugins.platforms.matrix.voice_mention import has_voice_marker


def classify_inbound_media(msgtype: str, event_mimetype: str, source_content: dict) -> tuple[MessageType, str, bool]:
    """Map a Matrix media msgtype to (MessageType, mime type, is_voice_message)."""
    if msgtype == "m.image":
        return MessageType.PHOTO, event_mimetype or "image/png", False
    if msgtype == "m.audio":
        is_voice = has_voice_marker(source_content)
        return (MessageType.VOICE if is_voice else MessageType.AUDIO), event_mimetype or "audio/ogg", is_voice
    if msgtype == "m.video":
        return MessageType.VIDEO, event_mimetype or "video/mp4", False
    return MessageType.DOCUMENT, event_mimetype or "application/octet-stream", False


def declared_media_filename(source_content: dict, body: str) -> str:
    """The name inbound media bytes must be cached under — never a caption (#135898)."""
    declared = source_content.get("filename")
    return declared if isinstance(declared, str) and declared.strip() else body
