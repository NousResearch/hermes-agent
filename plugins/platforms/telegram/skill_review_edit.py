"""Explicit skill replacement replies, bound to actor, topic, request and revision."""
import asyncio
import logging

from plugins.platforms.telegram.memory_review_edit import _chat_key
from plugins.platforms.telegram.memory_review import _home_scope
from tools.skill_review import inspect, revise
from tools.skill_prompt_send import send_prompt

logger = logging.getLogger(__name__)
PREFIX = '✏️ Edit skill file · '


def states(adapter):
    if not hasattr(adapter, '_skill_review_edits'):
        adapter._skill_review_edits = {}
    return adapter._skill_review_edits


def edit_home(adapter, query, pending_id, revision, index):
    state = states(adapter).get(_chat_key(query.message))
    if (state and (state['id'], state['revision'], state['index']) == (pending_id, revision, index)
            and str(query.from_user.id) == state['user']
            and str(query.message.message_id) == state['instruction']):
        return state['home']
    return None


def scoped(home, fn, *args):
    with _home_scope(home):
        return fn(*args)


async def cancel_edit(adapter, query, pending_id, revision, index):
    key = _chat_key(query.message)
    state = states(adapter).get(key)
    if not state or (state['id'], state['revision'], state['index']) != (pending_id, revision, index):
        return False
    if str(query.from_user.id) != state['user']:
        return False
    states(adapter).pop(key, None)
    await adapter._bot.edit_message_text(chat_id=key[0], message_id=int(state['instruction']),
        text='Edit cancelled. The proposal is unchanged.', parse_mode=None, reply_markup=None)
    return True


async def begin_edit(adapter, query, home, pending_id, revision, index, extra, action):
    record = await asyncio.to_thread(scoped, home, inspect, pending_id)
    if not record or record.get('revision') != revision or not record.get('skill_review'):
        await query.answer(text='This proposal cannot be edited. Open its latest review.')
        return
    file = record['skill_review']['files'][index]
    if file.get('delete_package'):
        await query.answer(text='Disapprove this deletion to keep the skill and request a replacement proposal.', show_alert=True)
        return
    await query.answer()
    key = _chat_key(query.message)
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    message = await adapter._send_control_message(key[0], PREFIX + file['path'] + '\n\n'
        'Reply to this message with the full replacement text, or attach a UTF-8 text file. '
        'For long files, use an attachment so Telegram cannot split your replacement. '
        'The revised comparison will be shown for approval. Reply Cancel to keep the current proposal.',
        parse_mode=None, thread_id=key[1], metadata={'notify': True},
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('Cancel edit',
            callback_data=f'skl:cancel:{pending_id}:{revision}:{index}')]]))
    states(adapter)[key] = dict(home=home, id=pending_id, revision=revision, index=index,
        user=str(query.from_user.id), instruction=str(message.message_id), query=query, record=record)


async def handle_edit_message(adapter, message):
    reply = getattr(message, 'reply_to_message', None)
    explicit = (reply is not None and str(getattr(getattr(reply, 'from_user', None), 'id', ''))
        == str(getattr(getattr(adapter, '_bot', None), 'id', ''))
        and (getattr(reply, 'text', '') or '').startswith(PREFIX))
    if not explicit:
        return False
    key = _chat_key(message)
    state = states(adapter).get(key)
    if (not state or str(getattr(message.from_user, 'id', '')) != state['user']
            or str(reply.message_id) != state['instruction']):
        await adapter._send_control_message(key[0], 'This edit request is no longer active. Tap Edit on the latest skill review.',
            parse_mode=None, thread_id=key[1], metadata={'notify': True})
        return True
    text = message.text or ''
    if text.strip().lower() == 'cancel':
        # The same actor/request checks protect textual cancellation.
        states(adapter).pop(key, None)
        await adapter._send_control_message(key[0], 'Edit cancelled. The proposal is unchanged.',
            parse_mode=None, thread_id=key[1], metadata={'notify': True})
        return True
    document = getattr(message, 'document', None)
    try:
        if document is not None:
            from tools.skill_manager_tool import MAX_SKILL_FILE_BYTES
            if document.file_size is None or document.file_size > MAX_SKILL_FILE_BYTES:
                raise ValueError('Attach a text file within the skill file size limit.')
            remote = await document.get_file()
            data = await remote.download_as_bytearray()
            if len(data) > MAX_SKILL_FILE_BYTES:
                raise ValueError('The attached file exceeds the skill file size limit.')
            text = data.decode('utf-8-sig')
        elif len(text.encode('utf-16-le')) // 2 >= 4000:
            # A split paste must never become an approved, truncated replacement.
            raise ValueError('Telegram may split this paste. Attach the complete replacement as a UTF-8 text file instead.')
        elif not text.strip():
            raise ValueError('Reply with replacement text or attach a UTF-8 text file.')
        record = await asyncio.to_thread(scoped, state['home'], revise,
            state['id'], state['revision'], state['index'], text)
    except (ValueError, OSError) as exc:
        await adapter._send_control_message(key[0], 'Replacement not accepted: ' + str(exc)[:700],
            parse_mode=None, thread_id=key[1], metadata={'notify': True})
        return True
    states(adapter).pop(key, None)
    try:
        await adapter._bot.edit_message_text(chat_id=key[0], message_id=int(state['instruction']),
            text='Replacement received. Review the revised skill proposal below.', parse_mode=None, reply_markup=None)
        from plugins.platforms.telegram.skill_review import status
        await status(adapter, state['query'], state['record'], state['index'], '✏️ Replaced by the revised proposal below')
    except Exception:
        logger.warning('Could not retire the old skill edit request; delivering the new review', exc_info=True)
    await asyncio.to_thread(scoped, state['home'], send_prompt, record)
    return True
