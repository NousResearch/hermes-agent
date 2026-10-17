"""A discussion recommendation becomes the exact pending draft only after a human tap."""
import copy
import json
import time
from types import SimpleNamespace

import pytest

from tests.gateway.test_memory_review_buttons import review as _review_fixture, click
from telegram import Update
from tests.gateway.test_memory_review_discussion import agent_for, memory, open_discussion
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tools import write_approval as wa
from tools.memory_tool import load_on_disk_store
from utils import atomic_json_write

review = _review_fixture


async def setup_discussion(review, kind='replace'):
    config = json.loads((review.home / 'config.yaml').read_text())
    config['memory'] = {'write_approval': True, 'user_char_limit': 16000}
    (review.home / 'config.yaml').write_text(json.dumps(config))
    store = load_on_disk_store()
    assert store.add('user', 'Original standing rule.')['success']
    assert store.add('user', 'Independent preference.')['success']
    payload = {'action': 'add' if kind == 'add' else 'replace', 'target': 'user', 'content': 'Initial proposal.'}
    if kind != 'add':
        payload.update(old_text='Original standing rule.', matched_entry='Original standing rule.')
    record = wa.stage_write('memory', payload, summary='', origin='test')
    record = wa.get_pending('memory', record['id'])
    event = await open_discussion(review, record)
    db = SessionDB(review.home / 'state.db')
    agent = agent_for(db)
    tokens = set_session_vars(platform='telegram', session_id=agent.session_id, user_id='111', chat_id='111',
                              message_id=event.source.message_id)
    args = dict(pending_id=record['id'], expected_revision=1)
    assert memory(agent, action='review', **args)['success']
    return record, agent, tokens, args


async def use_draft(review, record, draft, *, action='use', user=111, chat=111, token=None, mid=None, bot_id=None, thread_id=None):
    receipt = draft['receipt']
    raw = review.server.callback(user, chat, int(mid or receipt['message_ids'][-1]),
        f"mem:{action}:{record['id']}:{record.get('revision', 1)}:{token or draft['token']}")
    if bot_id is not None:
        raw['callback_query']['message']['from']['id'] = bot_id
    if thread_id is not None:
        raw['callback_query']['message']['message_thread_id'] = thread_id
    await review.adapter._handle_callback_query(Update.de_json(raw, review.bot), None)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['replace', 'remove', 'add', 'long_unicode', 'delivery_failed', 'selection_delivery_failed'])
async def test_use_exact_displayed_draft_then_approve_with_restart_and_profile_scope(review, tmp_path, kind):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    record, agent, tokens, args = await setup_discussion(review, kind)
    try:
        from agent.reply_delivery import consume_reply_delivery
        agent._current_turn_id = 'suggestion-turn'
        before = load_on_disk_store()._path_for('user').read_bytes()
        content = 'Durable rule: restart every worker; controller LAST. 🦋 <literal> `code`'
        if kind == 'long_unicode':
            content = (content + '\n' + ('Unicode 🦋 and literal *data*. ' * 170)).strip()
        kwargs = {'remove_entry': True} if kind == 'remove' else {'content': content}
        if kind == 'delivery_failed':
            config_text = (review.home / 'config.yaml').read_text()
            config = json.loads(config_text)
            config.pop('platforms')
            (review.home / 'config.yaml').write_text(json.dumps(config))
            failed = memory(agent, action='suggest', explanation='Keep the useful rule.', **kwargs, **args)
            assert failed['success'] and not failed['delivered']
            assert not consume_reply_delivery(agent, agent._current_turn_id)
            current = wa.get_pending('memory', record['id'])
            assert not current['discussion_draft'].get('receipt')
            assert current['payload'] == record['payload'] and current['revision'] == 1
            assert current['last_prompt_at'] == record['last_prompt_at']
            (review.home / 'config.yaml').write_text(config_text)
        sent_before = len(review.server.calls_of('sendMessage'))
        suggestion = memory(agent, action='suggest', explanation='Keep the standing rule; drop the incident details.', **kwargs, **args)
        assert suggestion['success'] and suggestion['delivered'] and not suggestion.get('staged')
        assert consume_reply_delivery(agent, agent._current_turn_id)
        assert not consume_reply_delivery(agent, agent._current_turn_id)
        from agent.reply_delivery import mark_reply_delivered
        mark_reply_delivered(agent)
        assert not consume_reply_delivery(agent, 'a-later-turn')
        current = wa.get_pending('memory', record['id'])
        draft = current['discussion_draft']
        assert current['payload'] == record['payload'] and current['revision'] == 1
        assert load_on_disk_store()._path_for('user').read_bytes() == before
        messages = [c.params for c in review.server.calls_of('sendMessage')[sent_before:]]
        # Every byte of the frozen candidate appears, in order, in native entities.
        displayed = ''.join(m['text'] for m in messages)
        assert ('Original standing rule.' if kind == 'remove' else content) in displayed
        assert messages[-1]['reply_markup']['inline_keyboard'][0][0]['text'] == 'Use this draft'
        assert all(not m.get('reply_markup') for m in messages[:-1])
        assert not messages[-1]['disable_notification']
        # A fresh process/agent needs no in-memory discussion state. The callback finds
        # its owner even when invoked outside that home's scope (A -> B -> A).
        other = tmp_path / 'other'
        other.mkdir()
        review.adapter.gateway_runner = SimpleNamespace(_served_profile_homes={'owner': review.home})
        if kind == 'selection_delivery_failed':
            config_text = (review.home / 'config.yaml').read_text()
            config = json.loads(config_text)
            config.pop('platforms')
            (review.home / 'config.yaml').write_text(json.dumps(config))
        token = set_hermes_home_override(other)
        try:
            await use_draft(review, record, draft)
            assert wa.list_pending('memory') == []
            assert load_on_disk_store().user_entries == []
        finally:
            reset_hermes_home_override(token)
        revised = wa.get_pending('memory', record['id'])
        if kind == 'selection_delivery_failed':
            assert not revised.get('prompt') and not revised.get('last_prompt_at')
            assert 'could not be delivered' in review.server.calls_of('sendMessage')[-1].params['text']
            (review.home / 'config.yaml').write_text(config_text)
            from tools.memory_prompt_send import send_prompt
            assert send_prompt('memory', revised)
            revised = wa.get_pending('memory', record['id'])
        assert revised['revision'] == 2
        assert revised['payload'] == draft['payload']
        assert load_on_disk_store()._path_for('user').read_bytes() == before
        assert 'discussion_draft' not in revised
        count = len(review.server.calls_of('sendMessage'))
        await use_draft(review, record, draft)  # double tap cannot revise again
        await click(review, record, 'approve')  # original card cannot save old wording
        assert len(review.server.calls_of('sendMessage')) == count
        assert load_on_disk_store()._path_for('user').read_bytes() == before
        await click(review, revised, 'approve')
        entries = load_on_disk_store().user_entries
        assert 'Independent preference.' in entries
        if kind == 'remove':
            assert entries == ['Independent preference.']
        else:
            assert content in entries and 'Initial proposal.' not in entries
            assert ('Original standing rule.' in entries) == (kind == 'add')
        assert wa.get_pending('memory', record['id']) is None
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize('conflict', ['superseded', 'actor', 'chat', 'topic', 'message', 'bot', 'token',
                                      'approve_on_draft', 'snoozed', 'rejected', 'original', 'capacity', 'no_history'])
async def test_draft_button_refuses_unreviewed_foreign_stale_or_unfitting_candidates(review, conflict):
    record, agent, tokens, args = await setup_discussion(review)
    try:
        before = load_on_disk_store()._path_for('user').read_bytes()
        if conflict == 'no_history':
            current = wa.get_pending('memory', record['id'])
            current['discussion'].pop('history_checked_revision')
            atomic_json_write(wa._pending_path('memory', record['id']), current)
            assert not memory(agent, action='suggest', content='Draft.', **args)['success']
            assert 'discussion_draft' not in wa.get_pending('memory', record['id'])
            return
        assert memory(agent, action='suggest', content='Exact draft.', explanation='Keep the useful rule.', **args)['delivered']
        current = wa.get_pending('memory', record['id'])
        draft = copy.deepcopy(current['discussion_draft'])
        kwargs = {}
        # health: allow HX011 -- Parameterized conflict injection; branches alter distinct approval inputs and share the refusal assertions.
        if conflict == 'superseded':
            assert memory(agent, action='suggest', content='A newer draft.', **args)['delivered']
        elif conflict in {'actor', 'chat', 'topic', 'message', 'bot', 'token', 'approve_on_draft'}:
            kwargs = {'actor': {'user': 222}, 'chat': {'chat': 222}, 'topic': {'thread_id': 7},
                      'message': {'mid': int(draft['receipt']['message_ids'][-1]) + 1000},
                      'bot': {'bot_id': 999}, 'token': {'token': '0' * 12},
                      'approve_on_draft': {'action': 'approve'}}[conflict]
        elif conflict == 'snoozed':
            current['snooze_until'] = time.time() + 3600
            atomic_json_write(wa._pending_path('memory', record['id']), current)
        elif conflict == 'rejected':
            await click(review, current, 'reject')
        elif conflict == 'original':
            assert load_on_disk_store().replace('user', 'Original standing rule.', 'Changed original.')['success']
            before = load_on_disk_store()._path_for('user').read_bytes()
        elif conflict == 'capacity':
            config = json.loads((review.home / 'config.yaml').read_text())
            config['memory']['user_char_limit'] = 10
            (review.home / 'config.yaml').write_text(json.dumps(config))
        await use_draft(review, record, draft, **kwargs)
        assert load_on_disk_store()._path_for('user').read_bytes() == before
        latest = wa.get_pending('memory', record['id'])
        assert latest is None if conflict == 'rejected' else latest['payload'] == record['payload'] and latest['revision'] == 1
        assert not review.server.calls_of('answerCallbackQuery')[-1].params.get('text', '').startswith('Draft ready')
    finally:
        clear_session_vars(tokens)
