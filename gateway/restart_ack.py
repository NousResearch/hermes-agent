"""Deliver the /restart acknowledgement before the gateway starts stopping (#65057)."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def send_restart_ack(adapter, event, ack: str) -> bool:
    """Send ``ack`` to the requester now and wait for the platform's answer; False = not delivered
    (the caller then returns it through the ordinary reply path)."""
    if adapter is None:
        return False
    metadata = {"thread_id": event.source.thread_id} if event.source.thread_id else None
    try:
        result = await asyncio.wait_for(
            adapter.send(event.source.chat_id, ack, reply_to=event.message_id, metadata=metadata), timeout=10)
    except Exception:  # any adapter failure: the reply path is the fallback, never a lost /restart
        logger.warning("Could not send the /restart acknowledgement before shutdown", exc_info=True)
        return False
    return bool(getattr(result, "success", False))
