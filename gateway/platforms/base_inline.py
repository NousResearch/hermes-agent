"""Inline gateway replies and their admission receipt."""

import logging

logger = logging.getLogger(__name__)


async def dispatch_inline_reply(adapter, event, *, log_cmd=None) -> None:
    """Call the handler and send its reply inline, with retry, threading and
    ephemeral deletion — no session lifecycle (active-session bypass paths)."""
    from gateway.platforms.base import (
        _thread_metadata_for_event, _reply_anchor_for_event, _mark_notify_metadata,
    )
    thread_meta = _thread_metadata_for_event(event)
    event._gateway_accepted = True
    response = await adapter._message_handler(event)
    text, eph_ttl = adapter._unwrap_ephemeral(response)
    if not text:
        return
    if log_cmd is not None:
        logger.info("[%s] Sending command '/%s' response (%d chars) to %s", adapter.name, log_cmd,
                    len(text), event.source.chat_id)
    result = await adapter._send_with_retry(
        chat_id=event.source.chat_id, content=text, reply_to=_reply_anchor_for_event(event),
        metadata=_mark_notify_metadata(thread_meta))
    if eph_ttl > 0 and result.success and result.message_id:
        adapter._schedule_ephemeral_delete(event.source.chat_id, result.message_id, eph_ttl)
