"""Real registry + Telegram SDK/HTTP skill review and replacement integration."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.gateway.test_memory_review_buttons import review as _review_fixture  # Real SDK fixture clears gateway stubs.
from telegram import Update
from tools import skill_manager_tool as sm, skill_review as backend, write_approval as wa
from tests.tools.test_skill_review import ORIGINAL, REPLACEMENT

review = _review_fixture


def setup(review, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: review.home)
    path = review.home / 'config.yaml'
    cfg = json.loads(path.read_text())
    cfg['skills'] = {'write_approval': False}
    path.write_text(json.dumps(cfg))
    assert json.loads(sm.skill_manage('create', 'sample', content=ORIGINAL))['success']
    cfg['skills']['write_approval'] = True
    path.write_text(json.dumps(cfg))


def stage():
    from model_tools import handle_function_call
    result = json.loads(handle_function_call('skill_manage', {'operations': [
        {'action': 'patch', 'name': 'sample', 'old_string': 'original rule', 'new_string': 'revised rule'},
        {'action': 'write_file', 'name': 'sample', 'file_path': 'references/guide.md', 'file_content': 'Guide 😀\n'}]}))
    assert result.get('staged'), result
    return wa.get_pending('skills', result['pending_id'])


async def click(review, record, index, action, user=111, chat=111):
    receipt = record['skill_prompts'][str(index)][-1]
    verb, *extra = action.split(':')
    data = f"skl:{verb}:{record['id']}:{record['revision']}:{index}" + (':' + extra[0] if extra else '')
    raw = review.server.callback(user, chat, int(receipt['message_ids'][-1]), data)
    await review.adapter._handle_callback_query(Update.de_json(raw, review.bot), None)


async def reply(review, text, document=False):
    state = next(iter(review.adapter._skill_review_edits.values()))
    raw = (review.server.dm_document(111, 'replacement.md', text.encode(), 'text/plain') if document
        else review.server.dm(111, text))
    raw['message']['reply_to_message'] = {'message_id': int(state['instruction']),
        'date': raw['message']['date'], 'chat': raw['message']['chat'],
        'from': {'id': review.bot.id, 'is_bot': True, 'first_name': 'Review'}, 'text': '✏️ Edit skill file · /test'}
    handler = review.adapter._handle_media_message if document else review.adapter._handle_text_message
    await handler(Update.de_json(raw, review.bot), None)


@pytest.mark.asyncio
@pytest.mark.parametrize('action', ['approve', 'reject', 'snooze', 'wrong_user', 'wrong_chat', 'profile'])
async def test_native_file_cards_and_receipt_bound_atomic_decisions(review, monkeypatch, action):
    setup(review, monkeypatch)
    record = stage()
    cards = review.server.calls_of('sendMessage')
    assert len(cards) == 2
    assert all(c.params['disable_notification'] is False for c in cards)
    assert str(review.home / 'skills/sample/SKILL.md') in cards[0].params['text']
    assert ORIGINAL in cards[0].params['text'] and REPLACEMENT in cards[0].params['text']
    assert [b['text'] for row in cards[0].params['reply_markup']['inline_keyboard'] for b in row] == [
        '✅ Approve', '🗑 Disapprove', '⏰ Snooze', '✏️ Edit']
    path = review.home / 'skills/sample/SKILL.md'
    if action in {'wrong_user', 'wrong_chat'}:
        await click(review, record, 0, 'approve', user=222 if action == 'wrong_user' else 111,
            chat=222 if action == 'wrong_chat' else 111)
        assert backend.inspect(record['id'])['skill_review']['approved'] == []
    elif action == 'snooze':
        await click(review, record, 0, 'snooze')
        await click(review, record, 0, 'snooze:1h')
        snoozed = backend.inspect(record['id'])
        assert snoozed['revision'] == 2 and snoozed['snooze_until'] > 0
        await click(review, record, 1, 'approve')
        assert backend.inspect(record['id'])['skill_review']['approved'] == []
    elif action == 'reject':
        await click(review, record, 0, 'approve')
        await click(review, record, 1, 'reject')
        assert wa.get_pending('skills', record['id']) is None
    else:
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        other = review.home / 'other'
        other.mkdir()
        review.adapter.gateway_runner = SimpleNamespace(_served_profile_homes={'owner': review.home})
        token = set_hermes_home_override(other) if action == 'profile' else None
        try:
            await click(review, record, 0, 'approve')
            assert path.read_text() == ORIGINAL
            await click(review, record, 1, 'approve')
            await click(review, record, 1, 'approve')
        finally:
            if token is not None:
                reset_hermes_home_override(token)
        assert path.read_text() == REPLACEMENT
        assert wa.get_pending('skills', record['id']) is None
        assert (review.home / 'skills/sample/references/guide.md').read_text() == 'Guide 😀\n'
        assert not (other / 'skills/sample').exists()
        return
    assert path.read_text() == ORIGINAL
    assert not (review.home / 'skills/sample/references/guide.md').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['text', 'attachment', 'long_paste', 'obsolete', 'command', 'retry'])
async def test_replacements_reopen_and_retry_preserve_full_file_review(review, monkeypatch, kind):
    setup(review, monkeypatch)
    record = stage()
    if kind == 'command':
        from gateway.slash_commands import GatewaySlashCommandsMixin
        from gateway.config import Platform
        from gateway.session import SessionSource
        from gateway.platforms.event import MessageEvent
        runner = SimpleNamespace(_delivery_adapter_for=lambda _: review.adapter, _write_approval_setter=lambda *a: None)
        source = SessionSource(platform=Platform.TELEGRAM, user_id='111', chat_id='111', chat_type='dm')
        for command in ['/skills', '/skills pending', '/skills diff ' + record['id']]:
            before = len(review.server.calls_of('sendMessage'))
            assert await GatewaySlashCommandsMixin._handle_skills_command(runner, MessageEvent(text=command, source=source)) == ''
            assert len(review.server.calls_of('sendMessage')) == before + 2
        return
    if kind == 'retry':
        from tools.skill_prompt_send import send_prompt
        # Fail only the second file after the first file's receipt was saved.
        review.server.fail('sendMessage', {'ok': False, 'error_code': 400, 'description': 'Test unavailable'},
            match=lambda p: '/references/guide.md' in p.get('text', ''))
        assert not send_prompt(record)
        failed = wa.get_pending('skills', record['id'])
        assert failed['skill_delivery_in_progress']['0']['complete']
        before = len(review.server.calls_of('sendMessage'))
        assert send_prompt(failed)
        assert len(review.server.calls_of('sendMessage')) == before + 1
        return
    await click(review, record, 0, 'approve')
    await click(review, record, 0, 'edit')
    replacement = REPLACEMENT + '\nHuman text 😀 <b>raw & whole</b>\n'
    if kind in {'attachment', 'long_paste'}:
        replacement += 'long content 😀\n' * 700
    if kind == 'obsolete':
        review.server.fail('editMessageText', {'ok': False, 'error_code': 400, 'description': 'message to edit not found'})
    await reply(review, replacement, document=kind == 'attachment')
    revised = backend.inspect(record['id'])
    if kind == 'long_paste':
        assert revised['revision'] == 1
        assert 'attach' in review.server.calls_of('sendMessage')[-1].params['text'].lower()
        return
    assert revised['revision'] == 2 and revised['skill_review']['approved'] == []
    assert revised['skill_review']['files'][0]['after'] == replacement
    assert (review.home / 'skills/sample/SKILL.md').read_text() == ORIGINAL
    cards = review.server.calls_of('sendMessage')
    assert all(len(c.params['text'].encode('utf-16-le')) // 2 <= 4096 for c in cards)
    rendered = ''.join(c.params['text'] for c in cards[-len(revised['skill_prompts']['0'][-1]['message_ids'])-1:-1])
    # Native entities, including quotations, never replace or escape the raw text.
    assert 'Human text 😀 <b>raw & whole</b>' in rendered
    from tools.skill_prompt_text import render_file
    quoted = []
    for card in render_file(revised, 0):
        raw_text = card.text.encode('utf-16-le')
        for entity in card.entities:
            if entity['type'] == 'blockquote':
                quoted.append(raw_text[2*entity['offset']:2*(entity['offset']+entity['length'])].decode('utf-16-le'))
    assert ''.join(quoted) == ORIGINAL + replacement
    await click(review, record, 1, 'approve')  # obsolete revision cannot apply
    assert (review.home / 'skills/sample/SKILL.md').read_text() == ORIGINAL
    await click(review, revised, 0, 'approve')
    await click(review, revised, 1, 'approve')
    assert (review.home / 'skills/sample/SKILL.md').read_text() == replacement
