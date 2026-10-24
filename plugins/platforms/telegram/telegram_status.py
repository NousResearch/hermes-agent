"""Serialized editable status bookkeeping for the Telegram adapter."""

import asyncio
from typing import TYPE_CHECKING, Any

from gateway.platforms.base import SendResult

if TYPE_CHECKING:
    from .adapter import TelegramAdapter


async def send_or_update_status(
    adapter: "TelegramAdapter", chat_id: str, status_key: str, content: str,
    *, metadata: dict[str, Any] | None = None,
) -> SendResult:
    """Coalesce one key without lending its cached message to overlapping sends."""
    key = (str(chat_id), str(status_key))
    lock = adapter._status_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        adapter._status_locks[key] = lock
    # Hold a strong reference while waiting AND editing/sending: weak values
    # release idle keys, never an active lock that another caller must share.
    async with lock:
        cached_id = adapter._status_message_ids.get(key)
        if cached_id is not None:
            result = await adapter.edit_message(chat_id, cached_id, content, finalize=True, metadata=metadata)
            if result.success:
                # Only write back if nobody evicted/replaced this key during the await.
                if result.message_id and adapter._status_message_ids.get(key) == cached_id:
                    adapter._status_message_ids[key] = str(result.message_id)
                return result
            adapter._status_message_ids.pop(key, None)
        result = await adapter.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            if len(adapter._status_message_ids) >= adapter._STATUS_MESSAGE_IDS_MAX:
                # FIFO trim: drop the oldest half to bound memory (mirrors the Slack adapter).
                for stale in list(adapter._status_message_ids)[: adapter._STATUS_MESSAGE_IDS_MAX // 2]:
                    adapter._status_message_ids.pop(stale, None)
            adapter._status_message_ids[key] = str(result.message_id)
        return result
