"""Telegram long-reply handoff, and the MarkdownV2 strip a split chunk's plain-text fallback uses.

A degenerate 415,742-char final once went out as 402 back-to-back ``sendMessage`` calls and held
the chat in multi-minute flood control (#152012). A reply that needs more than the adapter's
``MAX_SPLIT_MESSAGES`` messages goes out whole as one ``.md`` file instead (the stream consumer
does the same for a streamed reply). The adapter facade imports these helpers so it stays under
its line cap.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Telegram's ``sendDocument`` cap is 50 MB; keep each reply file well inside it.
REPLY_FILE_MAX_BYTES = 45 * 1024 * 1024


def reply_message_count(adapter: Any, content: str, metadata: Optional[dict[str, Any]]) -> int:
    """Messages ``send(content)`` would post: its rich parts when it goes out rich, else its
    MarkdownV2 chunks (escaping can push a raw-sized text past 4,096 UTF-16 units)."""
    from gateway.platforms.base import utf16_len
    from plugins.platforms.telegram.rich_parts import rich_part_count
    parts = rich_part_count(adapter, content, metadata)
    if parts is not None:
        return parts
    limit = adapter.MAX_MESSAGE_LENGTH
    formatted = adapter.format_message(content)
    size = utf16_len(formatted)
    if size <= limit:
        return 1
    return len(adapter.truncate_message(formatted, limit, len_fn=utf16_len))


async def send_reply_document(adapter: Any, chat_id: str, file_path: str, caption: Optional[str] = None,
                              file_name: Optional[str] = None, reply_to: Optional[str] = None,
                              metadata: Optional[dict[str, Any]] = None) -> Any:
    """``sendDocument`` for a long reply file whose failure is a failed result (``send_document``
    would post a text notice and report it sent). A short flood wait is retried inline; a longer
    one fails, and the deferred re-upload it schedules does nothing: the reply has already gone out
    as messages by then."""
    from gateway.platforms.base import SendResult
    from plugins.platforms.telegram.telegram_media_flood import _FLOOD_STAGE

    def build(handle: Any) -> dict[str, Any]:
        return {"document": handle, "filename": file_name or file_path.rsplit("/", 1)[-1],
                "caption": adapter._caption_1024(caption)}

    async def failed(error: Exception) -> Any:
        return SendResult(success=False, error=str(error))

    async def redo(path: Optional[str]) -> Any:
        if _FLOOD_STAGE.get() >= 2:  # the deferred re-upload: the messages fallback already delivered it
            return SendResult(success=False, error="long reply file superseded by messages")
        return await adapter._send_local_file("File", path or file_path, chat_id, reply_to, metadata, "document",
                                              build, failed, redo=redo, source=file_path)
    return await redo(None)


def _strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 escapes and formatting markers for the plain-text fallback."""
    cleaned = re.sub(r'\\([_*\[\]()~`>#\+\-=|{}.!\\])', r'\1', text)  # escape backslashes
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **bold** BEFORE MarkdownV2 *bold*
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)  # italic; word-bounded so snake_case survives
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)  # strikethrough
    cleaned = re.sub(r'\|\|([^|]+)\|\|', r'\1', cleaned)  # spoiler
    return cleaned
