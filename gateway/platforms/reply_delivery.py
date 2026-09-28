"""Sequential final-reply delivery with the ledger tracking only the unsent tail."""
import asyncio


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
