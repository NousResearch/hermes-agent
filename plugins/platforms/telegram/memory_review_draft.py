"""Select the exact discussion draft for a fresh independent approval."""
import asyncio
import logging
import time

from tools import write_approval as wa
from tools.memory_prompt_send import record_lock
from tools.memory_review_discussion import _budget, _same_actor, present_revision, stage_revision

logger = logging.getLogger(__name__)


def select_draft(home, pending_id, revision, token, actor):
    from plugins.platforms.telegram.memory_review import _home_scope
    from tools.memory_tool import load_on_disk_store

    with _home_scope(home):
        with record_lock(pending_id):
            record = wa.get_pending(wa.MEMORY, pending_id)
            d = (record or {}).get('discussion') or {}
            draft = (record or {}).get('discussion_draft') or {}
            if (not record or record.get('revision', 1) != revision or draft.get('revision') != revision
                    or draft.get('token') != token or not draft.get('receipt')
                    or not d.get('active') or d.get('history_checked_revision') != revision
                    or d.get('revision') != revision or not _same_actor(d, actor)
                    or d.get('session_id') != draft.get('context', {}).get('session_id')
                    or record.get('snooze_until', 0) > time.time()):
                return {'success': False, 'error': 'This draft changed or belongs to another discussion. Open the latest card.'}
            store = load_on_disk_store()
            result = store.preview_operation(draft['payload'])
            if not result.get('success'):
                return {'success': False, 'error': result.get('error', 'This draft no longer fits. Chat about it again.')}
            budget = _budget(record, store)
            stage_revision(record, draft['payload'], draft['context'])
        return present_revision(record, budget)


async def use_draft(adapter, query, cb, home, pending_id, action, extra):
    actor = {'platform': 'telegram', 'user_id': str(query.from_user.id), 'chat_id': str(query.message.chat_id),
             'thread_id': str(query.message.message_thread_id or '')}
    # A callback is a human selection, not another agent turn or a memory write.
    await query.answer(text='Preparing the review…')
    result = await asyncio.to_thread(select_draft, home, pending_id, int(extra[0]), extra[1], actor)
    if not result['success']:
        detail = result['error']
        await adapter._send_control_message(actor['chat_id'], detail if len(detail) <= 320 else detail[:320] + '…',
            parse_mode=None, thread_id=actor['thread_id'] or None, metadata={'notify': True})
        return
    from telegram.error import TelegramError
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except TelegramError:
        # Promotion already succeeded. This obsolete button is rejected by its
        # revision; failing to remove it must not hide the delivery outcome.
        logger.warning('Could not remove an obsolete memory draft button for %s', pending_id, exc_info=True)
    if not result['delivered']:
        await adapter._send_control_message(actor['chat_id'],
            'Your draft is selected, but the approval card could not be delivered. It will retry. Memory is unchanged.',
            parse_mode=None, thread_id=actor['thread_id'] or None, metadata={'notify': True})
