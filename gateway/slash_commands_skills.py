"""Telegram /skills opens complete native file reviews through the shared backend."""
import asyncio
import time

from gateway.config import HomeChannel, Platform
from tools import write_approval as wa


def _open_reviews(delivery, pending_id):
    from tools.skill_prompt_send import send_prompt
    from tools.skill_review import inspect
    records = [wa.get_pending(wa.SKILLS, pending_id)] if pending_id else wa.list_pending(wa.SKILLS)
    if pending_id and not records[0]:
        return 'That skill proposal was not found.'
    failed = 0
    for record in records:
        if record.get('snooze_until', 0) > time.time():
            continue
        record = inspect(record['id'])
        if not send_prompt(record, delivery=delivery, due_before=time.time(), nag_seconds=0):
            failed += wa.get_pending(wa.SKILLS, record['id']) is not None
    return 'Some skill reviews could not be opened. Your proposals are still saved; try /skills again.' if failed else ''


async def open_skill_reviews(runner, event, pending_id=None):
    adapter = runner._delivery_adapter_for(event.source)
    if adapter is None or not getattr(adapter, '_bot', None):
        return 'Skill reviews could not be opened because Telegram is unavailable. Your proposals are still saved.'
    destination = HomeChannel(platform=Platform.TELEGRAM, chat_id=str(event.source.chat_id),
        thread_id=event.source.thread_id, name='Skill review')
    return await asyncio.to_thread(_open_reviews, (adapter.config, destination), pending_id)
