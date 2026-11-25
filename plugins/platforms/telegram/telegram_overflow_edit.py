"""An oversized final/preview edit split across the existing Telegram message plus continuations.

Moved out of the adapter facade (line cap). Every chunk that lands in a final is indexed in the
sent-text store as it lands: Telegram echoes no body in ``reply_to_message``, so a later
quote-reply to any chunk resolves only through that index (#147287).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from gateway.platforms.base import SendResult, utf16_len
from plugins.platforms.telegram.telegram_chunking import (
    separate_chunk_indicator_from_fence, split_raw_carrying_emphasis)
from plugins.platforms.telegram.telegram_split import quotable_text, _strip_mdv2

# The adapter's logger: overflow lines keep the name operators and tests filter on.
logger = logging.getLogger("plugins.platforms.telegram.adapter")


async def edit_overflow_split(
    adapter: Any, chat_id: str, message_id: str, content: str, *, finalize: bool, metadata: Optional[dict[str, Any]] = None,
) -> SendResult:
    """Split an oversized edit across the existing message + continuations: edit ``message_id`` with
    chunk 1, send the rest as replies to the previous chunk, return ``message_id=<last-chunk-id>`` so
    the consumer keeps editing the newest message. ``success=False`` only if the first-chunk edit fails."""
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text

    raw_chunks, chunks = split_raw_carrying_emphasis(
        content, adapter.MAX_MESSAGE_LENGTH, adapter.truncate_message, utf16_len,
        formatted_len=(lambda chunk: utf16_len(adapter.format_message(chunk))) if finalize else None)
    if len(chunks) <= 1:
        chunks = [content]  # defensive: a single chunk just edits normally
    first_chunk = chunks[0]
    try:
        if finalize:
            await adapter._edit_markdown_or_plain(
                chat_id, message_id, separate_chunk_indicator_from_fence(adapter.format_message(first_chunk)), _strip_mdv2(first_chunk),
                "[%s] Overflow split: MarkdownV2 first-chunk edit failed, falling back to plain text: %s")
        else:
            await adapter._edit_text(chat_id, message_id, first_chunk)
    except Exception as e:
        if "not modified" not in str(e).lower():  # identical first chunk still sends continuations
            logger.error("[%s] Overflow split: first-chunk edit failed: %s", adapter.name, _redact_telegram_error_text(e), exc_info=True)
            return SendResult(success=False, error=_redact_telegram_error_text(e))
    if finalize:
        await adapter._record_sent_text(chat_id, message_id, quotable_text(first_chunk))
    # Continuations call adapter._bot.send_message directly to skip adapter.send's pre-chunking.
    continuation_ids: list[str] = []
    delivered_chunks = [first_chunk]
    prev_id = message_id
    thread_id = adapter._metadata_thread_id(metadata)
    for chunk in chunks[1:]:
        reply_to_id = int(prev_id) if prev_id else None
        thread_kwargs = adapter._thread_kwargs_for_send(chat_id, thread_id, metadata, reply_to_message_id=reply_to_id)
        sent_msg = await adapter._send_overflow_continuation(chat_id, chunk, reply_to_id, thread_kwargs, thread_id, metadata, finalize)
        flood = sent_msg if isinstance(sent_msg, SendResult) else None  # flood-refused: the caller holds off
        if sent_msg is None or flood:
            # Partial delivery: do NOT report success — the consumer would treat it as final delivery.
            logger.warning("[%s] Overflow split: stopped at %d/%d chunks delivered", adapter.name, 1 + len(continuation_ids), len(chunks))
            delivered_prefix = "".join(re.sub(r" \(\d+/\d+\)$", "", delivered) for delivered in raw_chunks[:len(delivered_chunks)])
            raw = {
                "partial_overflow": True, "delivered_chunks": 1 + len(continuation_ids),
                "total_chunks": len(chunks), "last_message_id": prev_id, "delivered_prefix": delivered_prefix,
                "continuation_message_ids": tuple(continuation_ids)}
            if flood and finalize:
                # A flood refusal posted nothing, so the rest is certain-undelivered: hand it over in the
                # shape _resume_partial_send sends (the MarkdownV2 continuation text), as send() does.
                raw["undelivered_chunks"] = tuple(
                    separate_chunk_indicator_from_fence(adapter.format_message(rest)) for rest in chunks[len(delivered_chunks):])
                raw["delivered_message_ids"] = (str(message_id), *continuation_ids)
            return SendResult(
                success=False, message_id=prev_id, retryable=True, retry_after=flood and flood.retry_after,
                error=flood.error if flood else "overflow_continuation_failed", raw_response=raw,
                continuation_message_ids=tuple(continuation_ids))
        new_id = str(getattr(sent_msg, "message_id", "")) or prev_id
        if finalize and new_id != prev_id:
            await adapter._record_sent_text(chat_id, new_id, quotable_text(chunk))
        continuation_ids.append(new_id)
        delivered_chunks.append(chunk)
        prev_id = new_id
    last_id = continuation_ids[-1] if continuation_ids else message_id
    logger.debug("[%s] Overflow split delivered %d chunks; last_id=%s", adapter.name, 1 + len(continuation_ids), last_id)
    return SendResult(success=True, message_id=last_id, continuation_message_ids=tuple(continuation_ids))
