"""Outbound status bubble helpers shared by the gateway turn progress path."""

from __future__ import annotations


async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status through adapter.send_or_update_status when supported (edits the previous
    bubble for the same status_key instead of appending); otherwise fall back to plain send.

    See #30045.
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)
