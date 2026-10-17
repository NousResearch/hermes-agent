"""Explicit replies to one memory's edit request, bound to user/topic/revision."""
from __future__ import annotations

import asyncio
import copy
import logging

logger = logging.getLogger(__name__)

from tools import write_approval as wa
from tools.memory_prompt_send import record_lock, review_buttons, send_prompt
from tools.memory_prompt_text import entry_key
from utils import atomic_json_write


def _chat_key(message):
    thread = getattr(message, 'message_thread_id', None)
    return str(message.chat_id), str(thread) if thread is not None else None


def _states(adapter):
    if not hasattr(adapter, '_memory_review_edits'):
        adapter._memory_review_edits = {}
    return adapter._memory_review_edits


def cancel_home(adapter, query, pending_id, revision):
    state = _states(adapter).get(_chat_key(query.message))
    if (state and state['id'] == pending_id and state['revision'] == revision
            and str(query.from_user.id) == state['user_id']
            and str(query.message.message_id) == state.get('instruction_id')):
        return state['home']
    return None


def _read(state):
    from plugins.platforms.telegram.memory_review import _home_scope
    with _home_scope(state['home']):
        return wa.get_pending(wa.MEMORY, state['id'])


async def _abandon(adapter, key, state):
    if _states(adapter).get(key) is state:
        _states(adapter).pop(key, None)
    record = await asyncio.to_thread(_read, state)
    if record and record.get('revision', 1) == state['revision']:
        await state['query'].edit_message_reply_markup(reply_markup=review_buttons(state['id'], state['revision']))
    if state.get('instruction_id'):
        await adapter._bot.edit_message_text(chat_id=key[0], message_id=int(state['instruction_id']),
            text='Edit cancelled. The proposal is unchanged.', parse_mode=None, reply_markup=None)


async def cancel_edit(adapter, query, pending_id):
    key = _chat_key(query.message)
    state = _states(adapter).get(key)
    if state and state['id'] == pending_id and str(query.from_user.id) == state['user_id']:
        await _abandon(adapter, key, state)
        return True
    return False


async def begin_edit(adapter, query, cb, home, pending_id, action, extra):
    key = _chat_key(query.message)
    previous = _states(adapter).get(key)
    if previous:
        await _abandon(adapter, key, previous)
    state = dict(home=home, id=pending_id, query=query, user_id=str(query.from_user.id), revision=int(extra[0]))
    record = await asyncio.to_thread(_read, state)
    if record is None or record.get('revision', 1) != state['revision'] or record['payload']['action'] == 'batch':
        await query.answer(text='Use the latest independent memory review.')
        return
    state['original_payload'] = copy.deepcopy(record['payload'])
    await query.answer()
    # Keep the original card usable if the gateway restarts during editing. Revision
    # validation protects an edit against a concurrent decision or another editor.
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    message = await adapter._send_control_message(key[0],
        f"✏️ Edit · {entry_key(record)}\n\nReply directly to this message with the complete replacement text. "
        "Only this memory will change. Your draft will be shown for approval before anything is saved.\n\n"
        "Reply Cancel to abandon this edit.", parse_mode=None, thread_id=key[1], metadata=None,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('Cancel edit',
            callback_data=f"mem:cancel:{pending_id}:{state['revision']}")]]))
    state['instruction_id'] = str(message.message_id)
    _states(adapter)[key] = state


def _save_edit(state, text):
    from plugins.platforms.telegram.memory_review import _home_scope
    from tools.memory_tool_store import _scan_memory_content
    with _home_scope(state['home']), record_lock(state['id']):
        record = wa.get_pending(wa.MEMORY, state['id'])
        if (record is None or record.get('revision', 1) != state['revision']
                or record['payload'] != state['original_payload']):
            return None, 'This proposal changed or was already handled. Reopen its latest review.'
        if error := _scan_memory_content(text):
            return None, 'The replacement was not accepted: ' + error
        payload = copy.deepcopy(record['payload'])
        payload['content'] = text
        if payload['action'] == 'remove':
            payload['action'] = 'replace'
        record['payload'] = payload
        record['action'] = payload['action']
        record['revision'] = state['revision'] + 1
        record.pop('prompt', None)
        record['prompts'] = []
        record.pop('last_prompt_at', None)
        record.pop('delivery_in_progress', None)
        record.pop('snooze_until', None)
        atomic_json_write(wa._pending_path(wa.MEMORY, state['id']), record)
        return record, None


def _deliver_edited(state, record):
    from plugins.platforms.telegram.memory_review import _home_scope
    with _home_scope(state['home']):
        result = send_prompt(wa.MEMORY, record)
        from tools.memory_space_refresh import refresh_space_reviews
        refresh_space_reviews()
        return result


async def _inactive_reply(adapter, message):
    reply = getattr(message, 'reply_to_message', None)
    bot = getattr(adapter, '_bot', None)
    if (reply is None or bot is None
            or str(getattr(getattr(reply, 'from_user', None), 'id', '')) != str(bot.id)
            or not (getattr(reply, 'text', '') or '').startswith('✏️ Edit · ')):
        return False
    # A restart or a completed capture must never turn an explicit replacement
    # reply into an ordinary agent turn. The user's text remains visible in chat.
    key = _chat_key(message)
    await adapter._send_control_message(key[0],
        'This edit request is no longer active. Your reply has not changed memory. '
        'Tap Edit on the latest review and reply to its new request.',
        parse_mode=None, thread_id=key[1], metadata=None)
    return True


async def handle_edit_message(adapter, message) -> bool:
    if not getattr(adapter, '_memory_review_edits', None):
        return await _inactive_reply(adapter, message)
    key = _chat_key(message)
    state = _states(adapter).get(key)
    if state is None or str(getattr(message.from_user, 'id', '')) != state['user_id']:
        return await _inactive_reply(adapter, message)
    text = message.text or ''
    if text.lstrip().startswith('/'):
        await _abandon(adapter, key, state)
        return False # the original command still reaches normal routing
    reply = getattr(message, 'reply_to_message', None)
    if reply is None or str(reply.message_id) != state.get('instruction_id'):
        return False
    if not text.strip() or text.strip().lower() == 'cancel':
        await _abandon(adapter, key, state)
        return True
    record, error = await asyncio.to_thread(_save_edit, state, text)
    if error:
        await adapter._send_control_message(key[0], error, parse_mode=None, thread_id=key[1], metadata=None)
        return True
    _states(adapter).pop(key, None)
    try:
        await adapter._bot.edit_message_text(chat_id=key[0], message_id=int(state['instruction_id']),
            text='✏️ Replacement received. Review the updated comparison before approving.', parse_mode=None, reply_markup=None)
        from plugins.platforms.telegram.memory_review import show_status
        original = {**record, 'payload': state['original_payload'], 'revision': state['revision']}
        await show_status(state['query'], original, '✏️ Replaced by your edited proposal below')
    except Exception:
        logger.warning('Could not update an obsolete edit request; the revised review will still be delivered', exc_info=True)
    await asyncio.to_thread(_deliver_edited, state, record)
    return True
