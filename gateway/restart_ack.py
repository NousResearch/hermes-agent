"""Deliver the /restart acknowledgement before the gateway starts stopping (#65057)."""

from __future__ import annotations

import asyncio
import logging

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, _mark_notify_metadata

logger = logging.getLogger(__name__)


async def send_restart_ack(runner, adapter, event, ack: str) -> bool:
    """Send ``ack`` to the requester now and wait for the platform's answer; False = not delivered
    (the caller then returns it through the ordinary reply path). Uses the ordinary command reply's
    anchor and routing metadata (Slack workspace, Telegram topic lane, notify)."""
    if adapter is None or adapter.platform is Platform.LOCAL:
        return False  # a LOCAL send is a no-op: the viewer reads the RETURNED reply (final_response)
    metadata = _mark_notify_metadata(runner._reply_metadata(event))
    try:
        result = await asyncio.wait_for(adapter.send(
            event.source.chat_id, ack, reply_to=runner._reply_anchor_for_event(event), metadata=metadata), timeout=10)
    except Exception:  # any adapter failure: the reply path is the fallback, never a lost /restart
        logger.warning("Could not send the /restart acknowledgement before shutdown", exc_info=True)
        return False
    # A read/write timeout may already have landed (``_send_with_retry`` does not resend it either):
    # count it as delivered so the reply path does not post a second copy. A timeout the adapter marks
    # as never sent (``retryable``, a connect timeout) still falls back, as ``_send_with_retry`` retries it.
    error = getattr(result, "error", None)
    never_sent = getattr(result, "retryable", False) or BasePlatformAdapter._is_retryable_error(error)
    return bool(getattr(result, "success", False)) or (not never_sent and BasePlatformAdapter._is_timeout_error(error))
