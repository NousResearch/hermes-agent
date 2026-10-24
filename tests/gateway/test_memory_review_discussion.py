"""An authorized review tap -> original history -> bounded draft -> independent approval."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_memory_review_buttons import review as _review_fixture, click
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_state import SessionDB
from tools import write_approval as wa
from tools.memory_tool import load_on_disk_store
from utils import atomic_json_write

review = _review_fixture


def agent_for(db, session_id='discussion'):
    return SimpleNamespace(session_id=session_id, _parent_session_id='', platform='telegram',
        _user_id='111', _chat_id='111', _thread_id='', model='test', provider='test',
        _session_db=db, _memory_store=load_on_disk_store(), _memory_manager=None)


def memory(agent, **args):
    return json.loads(INLINE_TOOL_EXECUTORS['memory'](agent, args, InlineToolContext('test', 'memory-call')))


async def open_discussion(review, record):
    accepted = asyncio.Event()
    events = []
    async def receive(event):
        events.append(event)
        accepted.set()
    review.adapter.set_message_handler(receive)
    await click(review, record, 'discuss')
    await asyncio.wait_for(accepted.wait(), 5)
    event, = events
    assert event.source.user_id == '111' and not event.source.is_bot
    assert event.source.chat_id == '111' and not event.internal
    assert event.allow_gateway_control is False
    assert event.metadata['memory_discussion']['id'] == record['id']
    assert json.loads(event.text.rsplit('\n', 1)[-1]) == record['payload']
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize('history_mode', ['source', 'background', 'legacy', 'quoted_legacy', 'missing', 'undeliverable'])
async def test_discussion_recovers_history_and_returns_one_size_checked_review(review, history_mode):
    config = json.loads((review.home / 'config.yaml').read_text())
    config['memory'] = {'write_approval': True, 'user_char_limit': 180}
    (review.home / 'config.yaml').write_text(json.dumps(config))
    db = SessionDB(review.home / 'state.db')
    db.create_session('creator', 'telegram')
    original = 'Release ordering: an old incident with redundant detail.'
    explanation = 'Release ordering: restart all workers, controller last, because it hosts the updater.'
    if history_mode == 'quoted_legacy':
        original = 'Release ordering ("ALL WORKERS RESTART CONTROLLER LAST"): an old incident.'
        explanation = 'ALL WORKERS RESTART after manual releases; CONTROLLER LAST because it hosts the updater.'
    if history_mode != 'missing':
        db.append_message('creator', 'user', explanation, platform_message_id='801')
        if history_mode != 'background':
            db.append_message('creator', 'assistant', tool_calls=[{'id': 'memory-call',
                'type': 'function', 'function': {'name': 'memory', 'arguments': '{}'}}])
    creator = agent_for(db, 'creator')
    if history_mode == 'background':
        # The real review fork shares the parent's session but disables persistence.
        creator._parent_session_id = 'creator'
        creator._session_db = None
    store = load_on_disk_store()
    assert store.add('user', original)['success']
    assert store.add('user', 'Keep other policy.')['success']
    result = memory(creator, action='patch', target='user', old_text=original,
                    content='Release ordering: restart the controller last.')
    record = wa.get_pending('memory', result['pending_id'])
    assert record['provenance']['session_id'] == 'creator'
    assert record['provenance']['tool_call_id'] == 'memory-call'
    if history_mode in {'legacy', 'quoted_legacy', 'missing'}:
        record.pop('provenance')
        atomic_json_write(wa._pending_path('memory', record['id']), record)
        db.create_session('earlier-discussion', 'telegram')
        db.append_message('earlier-discussion', 'user',
                          'Chat about this proposed memory change to USER.md.\n' + original)
        db.create_session('compressed-retelling', 'telegram')
        db.append_message('compressed-retelling', 'user',
                          '[CONTEXT COMPACTION — REFERENCE ONLY]\n' + original)
    first_event = await open_discussion(review, record)
    reopened = await open_discussion(review, record)
    assert reopened.message_id != first_event.message_id
    assert reopened.source.message_id == first_event.source.message_id == record['prompt']['message_ids'][-1]
    assert reopened.reply_anchor_override == record['prompt']['message_ids'][-1]
    before = store._path_for('user').read_bytes()
    agent = agent_for(db)
    tokens = set_session_vars(platform='telegram', session_id='discussion', user_id='111', chat_id='111',
                              message_id=record['prompt']['message_ids'][-1])
    try:
        args = dict(pending_id=record['id'], expected_revision=1)
        assert not memory(agent, action='add', target='user', content='duplicate before history')['success']
        assert not memory(agent, action='revise', content='premature draft', **args)['success']
        context = memory(agent, action='review', **args)
        assert context['success']
        history = context['history']
        assert 'earlier-discussion' not in json.dumps(history['exchanges'])
        assert 'compressed-retelling' not in json.dumps(history['exchanges'])
        if history_mode == 'source':
            assert history['status'] == 'source_anchor'
        if history_mode == 'background':
            assert history['status'] == 'source_session'
        if history_mode != 'missing':
            assert explanation in json.dumps(history)
            if history_mode in {'legacy', 'quoted_legacy'}:
                assert sum(bool(e['result'].get('results')) for e in history['exchanges']) == 1
                assert history['exchanges'][-1]['roles'] == 'user'
        else:
            assert history['status'] == 'not_found'
        assert context['budget']['maximum_draft_characters'] == 180 - len('Keep other policy.') - 3
        file_context = context['memory_files']['user']
        assert file_context['entries'] == [original, 'Keep other policy.']
        assert file_context['path'] == context['target_path']
        assert file_context['current_characters'] == len('\n§\n'.join(file_context['entries']))
        assert file_context['limit'] == context['budget']['limit']
        assert not memory(agent, action='add', target='user', content='duplicate')['success']
        assert not memory(agent, action='revise', content='unrequested draft', **args)['success']
        clear_session_vars(tokens)
        tokens = set_session_vars(platform='telegram', session_id='discussion', user_id='111', chat_id='111', message_id='902')
        oversized = memory(agent, action='revise', content='x' * 181, **args)
        assert not oversized['success'] and oversized['capacity']['over'] > 0
        assert wa.get_pending('memory', record['id'])['revision'] == 1
        assert store._path_for('user').read_bytes() == before
        # All association lives on disk: a fresh agent after a restart can finish the discussion.
        agent = agent_for(db)
        draft = 'Release ordering: restart all workers; controller last so the updater can finish.'
        if history_mode == 'undeliverable':
            config.pop('platforms')
            (review.home / 'config.yaml').write_text(json.dumps(config))
        revised = memory(agent, action='revise', content=draft, **args)
        assert revised['success'] and revised['staged']
        assert revised['pending_id'] == record['id'] and revised['revision'] == 2
        assert len(wa.list_pending('memory')) == 1 and store._path_for('user').read_bytes() == before
        current = wa.get_pending('memory', record['id'])
        assert current['payload']['matched_entry'] == original
        assert current['payload']['content'] == draft
        if history_mode == 'undeliverable':
            assert not revised['delivered'] and 'delivery failed' in revised['message']
            assert not current.get('prompt') and not current.get('last_prompt_at')
            return
        assert revised['delivered']
        keyboard = review.server.calls_of('sendMessage')[-1].params['reply_markup']['inline_keyboard']
        assert [len(row) for row in keyboard] == [2, 2, 1]
        await click(review, record, 'approve')
        assert store._path_for('user').read_bytes() == before
        await click(review, current, 'approve')
        assert load_on_disk_store().user_entries == [draft, 'Keep other policy.']
    finally:
        clear_session_vars(tokens)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('existing', [True, False])
async def test_discussion_can_propose_removing_only_its_existing_entry(review, existing):
    config = json.loads((review.home / 'config.yaml').read_text())
    config['memory'] = {'write_approval': True}
    (review.home / 'config.yaml').write_text(json.dumps(config))
    db = SessionDB(review.home / 'state.db')
    db.create_session('creator', 'telegram')
    db.append_message('creator', 'user', 'Lunch receipt: paid $18 once.')
    store = load_on_disk_store()
    original = 'Lunch receipt: paid $18 once.'
    if existing:
        store.add('user', original)
    store.add('user', 'Important standing preference.')
    agent = agent_for(db, 'creator')
    result = memory(agent, action='patch' if existing else 'add', target='user',
                    old_text=original if existing else None, content='Lunch receipt: $18.')
    record = wa.get_pending('memory', result['pending_id'])
    await open_discussion(review, record)
    agent = agent_for(db)
    tokens = set_session_vars(platform='telegram', session_id='discussion', user_id='111', chat_id='111',
                              message_id=record['prompt']['message_ids'][-1])
    try:
        args = dict(pending_id=record['id'], expected_revision=1)
        assert not memory(agent, action='revise', remove_entry=True, **args)['success']
        assert memory(agent, action='review', **args)['success']
        before = store._path_for('user').read_bytes()
        assert not memory(agent, action='revise', remove_entry=True, **args)['success']
        clear_session_vars(tokens)
        tokens = set_session_vars(platform='telegram', session_id='discussion', user_id='111', chat_id='111', message_id='902')
        assert not memory(agent, action='revise', remove_entry='not-a-bool', **args)['success']
        assert not memory(agent, action='revise', remove_entry=True, content='ambiguous', **args)['success']
        revised = memory(agent, action='revise', remove_entry=True, **args)
        assert store._path_for('user').read_bytes() == before
        if not existing:
            assert not revised['success']
            assert wa.get_pending('memory', record['id'])['revision'] == 1
            return
        assert revised['success'] and revised['delivered']
        current = wa.get_pending('memory', record['id'])
        assert current['payload']['action'] == 'remove'
        assert current['payload']['matched_entry'] == original
        assert current['revision'] == 2 and len(wa.list_pending('memory')) == 1
        card = review.server.calls_of('sendMessage')[-1].params['text']
        assert '\nDelete\n' in card and original in card and 'Replacement' not in card
        await click(review, record, 'approve')
        assert store._path_for('user').read_bytes() == before
        await click(review, current, 'approve')
        assert load_on_disk_store().user_entries == ['Important standing preference.']
    finally:
        clear_session_vars(tokens)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('conflict', ['user', 'chat', 'topic', 'session', 'revision', 'original', 'snooze', 'rejected', 'foreign_db'])
async def test_discussion_cannot_revise_another_actor_or_a_changed_review(review, tmp_path, conflict):
    config = json.loads((review.home / 'config.yaml').read_text())
    config['memory'] = {'write_approval': True}
    (review.home / 'config.yaml').write_text(json.dumps(config))
    db = SessionDB(review.home / 'state.db')
    db.create_session('creator', 'telegram')
    db.append_message('creator', 'user', 'Release ordering: keep all workers available.')
    store = load_on_disk_store()
    original = 'Release ordering: original policy.'
    store.add('user', original)
    creator = agent_for(db, 'creator')
    result = memory(creator, action='patch', target='user', old_text=original, content='Release ordering: revised policy.')
    record = wa.get_pending('memory', result['pending_id'])
    # Unauthorized taps cannot create the discussion or start an agent turn.
    await click(review, record, 'discuss', user=222)
    assert 'discussion' not in wa.get_pending('memory', record['id'])
    await open_discussion(review, record)
    agent = agent_for(db)
    tokens = set_session_vars(platform='telegram', session_id='discussion', user_id='111', chat_id='111',
                              message_id=record['prompt']['message_ids'][-1])
    foreign = None
    try:
        args = dict(pending_id=record['id'], expected_revision=1)
        assert memory(agent, action='review', **args)['success']
        clear_session_vars(tokens)
        tokens = set_session_vars(platform='telegram', session_id='discussion', user_id='111', chat_id='111', message_id='902')
        # health: allow HX011 -- Parameterized conflict injection; branches alter distinct approval inputs and share the refusal assertions.
        if conflict in {'user', 'chat', 'topic', 'session'}:
            setattr(agent, {'user': '_user_id', 'chat': '_chat_id', 'topic': '_thread_id', 'session': 'session_id'}[conflict], 'different')
        elif conflict == 'original':
            store.replace('user', original, 'Externally changed original')
        elif conflict == 'rejected':
            wa.discard_pending('memory', record['id'])
        elif conflict in {'revision', 'snooze'}:
            current = wa.get_pending('memory', record['id'])
            current['revision'] += 1
            if conflict == 'snooze': current['snooze_until'] = 9999999999
            atomic_json_write(wa._pending_path('memory', record['id']), current)
        else:
            foreign = SessionDB(tmp_path / 'foreign.db')
            agent._session_db = foreign
        before = store._path_for('user').read_bytes()
        before_record = wa.get_pending('memory', record['id'])
        action = 'review' if conflict == 'foreign_db' else 'revise'
        failed = memory(agent, action=action, content='must not overwrite', **args)
        assert not failed['success']
        assert store._path_for('user').read_bytes() == before
        assert wa.get_pending('memory', record['id']) == before_record
    finally:
        clear_session_vars(tokens)
        if foreign: foreign.close()
        db.close()
