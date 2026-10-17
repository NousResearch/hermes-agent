"""Native skill-file decisions bound to exact delivered bot/chat/topic/revision receipts."""
import asyncio
import logging
import re
import time
from pathlib import Path

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.telegram.memory_review import _home_scope, _SNOOZE_SECONDS
from tools import write_approval as wa
from tools.skill_prompt_text import buttons
from tools.skill_review import decide, review_lock, save

logger = logging.getLogger(__name__)


def find_home(adapter, query, pending_id, revision, index):
    homes = [get_hermes_home(), *(getattr(getattr(adapter, 'gateway_runner', None), '_served_profile_homes', {}) or {}).values()]
    for home in dict.fromkeys(Path(h) for h in homes):
        scope = set_hermes_home_override(home)
        try:
            record = wa.get_pending(wa.SKILLS, pending_id)
        finally:
            reset_hermes_home_override(scope)
        if not record or record.get('revision', 1) != revision or record.get('snooze_until', 0) > time.time():
            continue
        message = query.message
        for receipt in record.get('skill_prompts', {}).get(str(index), []):
            if (receipt.get('revision') == revision and receipt.get('chat_id') == str(message.chat_id)
                    and receipt.get('message_ids', [])[-1:] == [str(message.message_id)]
                    and receipt.get('bot_id') == str(getattr(message.from_user, 'id', ''))
                    and receipt.get('thread_id') == (str(message.message_thread_id) if message.message_thread_id is not None else None)):
                return home
    return None


async def notify(adapter, query, text):
    await adapter._send_control_message(str(query.message.chat_id), text, parse_mode=None,
        thread_id=str(query.message.message_thread_id) if query.message.message_thread_id else None,
        metadata={'notify': True})


async def status(adapter, query, record, index, text, *, active=False):
    from telegram.error import TelegramError
    from tools.memory_prompt_text import ReviewMessage
    receipts = record.get('skill_prompts', {}).get(str(index), [])
    receipt = next((r for r in receipts if r['message_ids'][-1:] == [str(query.message.message_id)]), None)
    if receipt is None:
        return
    data = receipt['card']
    card = ReviewMessage(data['text'], list(data['entities'])).append('\n\n' + text, 'bold')
    try:
        await query.edit_message_text(text=card.text, entities=card.entities, parse_mode=None,
            reply_markup=buttons(record['id'], record.get('revision', 1), index) if active else None)
    except TelegramError:
        logger.warning('Could not update obsolete skill review %s', record['id'], exc_info=True)
        await notify(adapter, query, text)


def _decide(home, pending_id, revision, index, action):
    with _home_scope(home):
        return decide(pending_id, revision, index, action)


async def decision(adapter, query, home, pending_id, revision, index, extra, action):
    await query.answer(text='Checking the reviewed files…')
    try:
        result = await asyncio.to_thread(_decide, home, pending_id, revision, index, action)
    except (ValueError, OSError) as exc:
        await notify(adapter, query, 'Could not apply this proposal. ' + str(exc)[:700])
        return
    await status(adapter, query, result['record'], index, result['status'])
    if result['finished']:
        # A transaction's older cards must visibly reflect its final outcome.
        from telegram.error import TelegramError
        for key, receipts in result['record'].get('skill_prompts', {}).items():
            for receipt in receipts:
                mid = receipt['message_ids'][-1]
                if str(mid) == str(query.message.message_id) and receipt['chat_id'] == str(query.message.chat_id):
                    continue
                from tools.memory_prompt_text import ReviewMessage
                data = receipt['card']
                card = ReviewMessage(data['text'], list(data['entities'])).append('\n\n' + result['status'], 'bold')
                try:
                    await adapter._bot.edit_message_text(chat_id=receipt['chat_id'], message_id=int(mid),
                        text=card.text, entities=card.entities, parse_mode=None, reply_markup=None)
                except TelegramError:
                    logger.warning('Could not retire a decided skill card %s', pending_id, exc_info=True)


def _snooze(home, pending_id, revision, duration):
    # Keep the same duration/timezone semantics without mutating the memory subsystem.
    from datetime import datetime, timedelta
    from hermes_time import get_timezone, safe_strftime
    with _home_scope(home), review_lock(pending_id):
        record = wa.get_pending(wa.SKILLS, pending_id)
        if not record or record.get('revision', 1) != revision:
            raise ValueError('This proposal changed. Open its latest review.')
        now = time.time()
        zone = get_timezone()
        until = ((datetime.fromtimestamp(now, zone) + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0).timestamp() if duration == 'tomorrow9' else now + _SNOOZE_SECONDS[duration])
        record['snooze_until'] = until
        record['revision'] = revision + 1
        record.pop('skill_delivery_in_progress', None)
        save(record)
        return record, '⏰ Snoozed until ' + safe_strftime(datetime.fromtimestamp(until, zone), '%Y-%m-%d %H:%M %Z')


async def snooze(adapter, query, home, pending_id, revision, index, extra, action):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    if not extra:
        await query.answer(text='How long would you like to snooze?')
        prefix = f'skl:snooze:{pending_id}:{revision}:{index}:'
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=prefix + value)
            for label, value in (('1h', '1h'), ('4h', '4h'), ('tomorrow 9am', 'tomorrow9'), ('3d', '3d'))],
            [InlineKeyboardButton('Cancel', callback_data=f'skl:cancel:{pending_id}:{revision}:{index}')]])
        await query.edit_message_reply_markup(reply_markup=markup)
        return
    await query.answer()
    try:
        record, text = await asyncio.to_thread(_snooze, home, pending_id, revision, extra[0])
    except (ValueError, OSError) as exc:
        await notify(adapter, query, str(exc))
        return
    await status(adapter, query, record, index, text)


async def cancel(adapter, query, home, pending_id, revision, index, extra, action):
    from plugins.platforms.telegram.skill_review_edit import cancel_edit
    await query.answer(text='Cancelled')
    if not await cancel_edit(adapter, query, pending_id, revision, index):
        await query.edit_message_reply_markup(reply_markup=buttons(pending_id, revision, index))


async def handle_callback(adapter, query, data, cb):
    if not await adapter._callback_authorized(query, cb, 'You are not authorized to review this skill.'):
        return
    parts = data.split(':')
    if (len(parts) not in {5, 6} or not re.fullmatch('[a-f0-9]{8}', parts[2])
            or not re.fullmatch('[1-9][0-9]{0,8}', parts[3]) or not re.fullmatch('[0-9]{1,5}', parts[4])):
        await query.answer(text='Invalid skill review action')
        return
    _, action, pending_id, rev, number, *extra = parts
    revision, index = int(rev), int(number)
    from plugins.platforms.telegram.skill_review_edit import begin_edit, edit_home
    handlers = {'approve': decision, 'reject': decision, 'snooze': snooze, 'cancel': cancel, 'edit': begin_edit}
    if action not in handlers or (extra and (action != 'snooze' or extra[0] not in {*_SNOOZE_SECONDS, 'tomorrow9'})):
        await query.answer(text='Invalid skill review action')
        return
    home = edit_home(adapter, query, pending_id, revision, index) if action == 'cancel' else None
    home = home or await asyncio.to_thread(find_home, adapter, query, pending_id, revision, index)
    if home is None:
        await query.answer(text='This review changed or was already handled. Open its latest card.')
        return
    await handlers[action](adapter, query, home, pending_id, revision, index, extra, action)
