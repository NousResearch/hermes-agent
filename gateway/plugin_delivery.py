"""Plugin observations of final-text transport acknowledgements (not model completion)."""
from __future__ import annotations

import json


async def report_delivery(adapter, event, session_key, result) -> None:
    from hermes_cli.lifecycle import ainvoke_hook, has_hook

    with adapter._media_delivery_scope(event.source):
        if not has_hook("post_message_delivered"):
            return
        message_ref = event.ledger_message_id or event.message_id
        # Durable automation uses a namespaced producer identity as the inbound id.
        # Decode only our own shape; ordinary platform IDs remain opaque.
        try:
            identity = json.loads(message_ref or "null")
        except (TypeError, ValueError):
            identity = None
        if (isinstance(identity, list) and len(identity) == 4
                and identity[:1] == ["plugin"] and identity[2] == session_key):
            message_ref = identity[3]
        await ainvoke_hook(
            "post_message_delivered", session_key=session_key, message_ref=message_ref,
            platform=event.source.platform.value, platform_message_id=result.message_id,
            success=bool(result.success), error=result.error,
        )


async def send_final(adapter, event, session_key, text_content, metadata, *,
                     reply_to, is_ephemeral_response=False):
    """Record the obligation before sending; finalize and notify only from the transport result.

    The current transport may have replaced the intake adapter during a reconnect. The returned
    adapter owns the message ID, including subsequent ephemeral deletion.
    """
    from gateway.platforms.base import logger, stop_reply_clock

    delivery_adapter = adapter._final_delivery_adapter(event.source)
    logger.info("[%s] Sending response (%d chars) to %s", delivery_adapter.name,
                len(text_content), event.source.chat_id)
    obligation_id = await adapter._record_delivery_obligation(
        event, session_key, text_content, delivery_adapter, is_ephemeral_response)
    if obligation_id is not None:
        await adapter._release_turn_marker(event)
    result = await delivery_adapter._send_with_retry(
        chat_id=event.source.chat_id, content=text_content, reply_to=reply_to, metadata=metadata)
    stop_reply_clock(delivery_adapter, event.source.chat_id, result)
    if obligation_id is not None:
        await adapter._finalize_delivery_obligation(obligation_id, result, event, delivery_adapter)
    await report_delivery(delivery_adapter, event, session_key, result)
    return result, delivery_adapter
