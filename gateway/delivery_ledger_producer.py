"""Record the final send before releasing its durable turn marker."""

import asyncio
import logging
from typing import Optional

from gateway.platforms.event import MessageEvent

logger = logging.getLogger(__name__)


async def record_delivery_obligation(
    source_adapter,
    event: MessageEvent,
    session_key: str,
    text_content: str,
    delivery_adapter,
    is_ephemeral_response: bool,
    routing_metadata: Optional[dict] = None,
) -> Optional[str]:
    """Ledger the final response BEFORE the send so a crash before platform ACK redelivers on
    next boot; best-effort, skips slash-command and ephemeral replies. Returns the obligation id
    or None."""
    if is_ephemeral_response or str(event.text or "").lstrip().startswith(
        ("/", source_adapter.typed_command_prefix or "!")
    ):
        return None
    try:
        from gateway.delivery_ledger import (
            compute_obligation_id,
            ledger_enabled,
            mark_attempting,
            record_obligation,
        )

        if not await asyncio.to_thread(ledger_enabled):
            return None
        source = event.source
        # ``ledger_message_id`` wins when set: a queued chain's final answers the last message
        # of the chain, not the event that opened it (see ``MessageEvent.ledger_message_id``).
        _ledger_id = getattr(event, "ledger_message_id", None)
        if _ledger_id is None:
            _ledger_id = getattr(event, "message_id", "")
        obligation_id = compute_obligation_id(
            session_key, str(_ledger_id or ""), text_content
        )
        await asyncio.to_thread(
            record_obligation,
            obligation_id=obligation_id,
            session_key=session_key,
            platform=str(getattr(source.platform, "value", source.platform)),
            chat_id=source.chat_id,
            thread_id=getattr(source, "thread_id", None),
            content=text_content,
            adapter_profile=getattr(delivery_adapter, "_owner_profile", None),
            routing_metadata=routing_metadata,
        )
        await asyncio.to_thread(mark_attempting, obligation_id)
        return obligation_id
    except Exception:
        logger.debug("delivery ledger record failed", exc_info=True)
        return None
