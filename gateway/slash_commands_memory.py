"""Telegram /memory review entry point, using the same cards as immediate staging."""
from __future__ import annotations

import asyncio
import time

from gateway.config import HomeChannel, Platform
from tools import write_approval as wa


def _open_reviews(delivery):
    from tools.memory_review_conversion import convert_pending_batches
    from tools.memory_prompt_send import send_prompt

    # A legacy batch must become independent durable decisions before it can be shown.
    convert_pending_batches()
    now = time.time()
    failed = 0
    for record in wa.list_pending(wa.MEMORY):
        if record.get('snooze_until', 0) > now:
            continue
        # Explicit opening ignores the reminder interval, but respects snooze and keeps
        # receipt/revision checks, grouping, and the durable reminder clock on one path.
        if not send_prompt(wa.MEMORY, record, due_before=now, nag_seconds=0, delivery=delivery):
            if wa.get_pending(wa.MEMORY, record['id']) is not None:
                failed += 1
    return ('Some memory reviews could not be opened. Your proposals are still saved; try /memory again.'
            if failed else '')


async def open_memory_reviews(runner, event):
    adapter = runner._delivery_adapter_for(event.source)
    if adapter is None or not getattr(adapter, '_bot', None):
        return 'Memory reviews could not be opened because Telegram is unavailable. Your proposals are still saved.'
    destination = HomeChannel(platform=Platform.TELEGRAM, chat_id=str(event.source.chat_id),
        thread_id=event.source.thread_id, name='Memory review')
    # The dispatcher binds the routed runtime home; to_thread preserves that scope.
    # The receiving transport and exact chat/topic come from the canonical delivery resolver.
    return await asyncio.to_thread(_open_reviews, (adapter.config, destination))
