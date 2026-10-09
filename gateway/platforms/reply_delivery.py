"""Sequential final-reply delivery with the ledger tracking only the unsent tail."""
import asyncio
import logging

from hermes_cli.observability.shared_metrics_gateway import stop_reply_clock

logger = logging.getLogger(__name__)


async def send_reply_chunks(adapter, chat_id, content, *, reply_to, metadata, obligation_id, split):
    from gateway.delivery_ledger import replace_remaining_content

    chunks = adapter.reply_chunks(content, chat_id) if split else [content]
    for index, chunk in enumerate(chunks):
        if index:
            await asyncio.sleep(1.0)
        result = await adapter._send_with_retry(
            chat_id=chat_id, content=chunk, reply_to=reply_to, metadata=metadata)
        if not result.success:
            return result
        if obligation_id is not None and index + 1 < len(chunks):
            # Checkpoint before the next send/sleep. Recovery may join the remaining
            # bubbles into one message, but must not replay the accepted prefix.
            await asyncio.to_thread(
                replace_remaining_content, obligation_id, "\n\n".join(chunks[index + 1:]))
    return result


async def send_final_ledgered(adapter, event, session_key, text_content, metadata, *,
                              reply_to, is_ephemeral_response=False, allow_reply_bursts=True):
    delivery_adapter = adapter._final_delivery_adapter(event.source)
    logger.info("[%s] Sending response (%d chars) to %s", delivery_adapter.name,
                len(text_content), event.source.chat_id)
    obligation_id = await adapter._record_delivery_obligation(
        event, session_key, text_content, delivery_adapter, is_ephemeral_response)
    marker_released = True
    if obligation_id is not None:
        marker_released = await adapter._release_turn_marker(event)  # ledger owns recovery
    result = await send_reply_chunks(
        delivery_adapter, event.source.chat_id, text_content,
        reply_to=reply_to, metadata=metadata, obligation_id=obligation_id,
        split=marker_released and allow_reply_bursts and not event.internal and not is_ephemeral_response and not str(event.text or "").lstrip().startswith(
            ("/", adapter.typed_command_prefix or "!")))
    stop_reply_clock(delivery_adapter, event.source.chat_id, result)
    if obligation_id is not None:
        await adapter._finalize_delivery_obligation(obligation_id, result, event, delivery_adapter)
    return result, delivery_adapter

