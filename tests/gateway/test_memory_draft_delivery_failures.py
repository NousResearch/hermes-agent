"""Discussion deliveries recheck ownership and survive obsolete-button failures."""
import json

import pytest

from tests.gateway.test_memory_discussion_draft import setup_discussion, use_draft
from tests.gateway.test_memory_review_buttons import review as _review_fixture, click, stage
from tests.gateway.test_memory_review_discussion import open_discussion
from gateway.session_context import clear_session_vars
from tools import write_approval as wa
from tools.memory_discussion_draft import send_draft
from tools.memory_review_context import source_context
from tools.memory_review_discussion import discussion_tool
from tools.memory_tool import load_on_disk_store

review = _review_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['another_discussion', 'snooze', 'edit', 'reject'])
async def test_draft_delivery_rechecks_discussion_after_registration(review, change):
    record, agent, tokens, args = await setup_discussion(review)
    try:
        before = load_on_disk_store()._path_for('user').read_bytes()
        # Exercise the real boundary between tool registration and native transport.
        result = discussion_tool('suggest', content='Frozen candidate.',
            context=source_context(agent), **args)
        assert result['success']
        if change == 'another_discussion':
            other = wa.get_pending('memory', stage('Another entry.')['id'])
            await open_discussion(review, other)
        elif change == 'snooze':
            await click(review, record, 'snooze:1h')
        elif change == 'edit':
            from plugins.platforms.telegram.memory_review_edit import _save_edit
            edited, error = _save_edit({'home': review.home, 'id': record['id'],
                'revision': 1, 'original_payload': record['payload']}, 'Human replacement.')
            assert edited and error is None
        else:
            await click(review, record, 'reject')
        sends = len(review.server.calls_of('sendMessage'))
        outcome = send_draft(result)
        assert not outcome['success']
        assert len(review.server.calls_of('sendMessage')) == sends
        assert load_on_disk_store()._path_for('user').read_bytes() == before
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize('approval_delivery_failed', [False, True])
async def test_selected_draft_is_handled_when_old_keyboard_cannot_be_removed(review, approval_delivery_failed):
    from tests.gateway.test_memory_review_discussion import memory
    record, agent, tokens, args = await setup_discussion(review)
    try:
        before = load_on_disk_store()._path_for('user').read_bytes()
        assert memory(agent, action='suggest', content='Selected exact draft.', **args)['delivered']
        draft = wa.get_pending('memory', record['id'])['discussion_draft']
        review.server.fail('editMessageReplyMarkup',
            {'ok': False, 'error_code': 400, 'description': 'Bad Request: message to edit not found'}, status=400)
        if approval_delivery_failed:
            config = json.loads((review.home / 'config.yaml').read_text())
            config.pop('platforms')
            (review.home / 'config.yaml').write_text(json.dumps(config))
        await use_draft(review, record, draft)
        selected = wa.get_pending('memory', record['id'])
        assert selected['revision'] == 2 and selected['payload'] == draft['payload']
        assert load_on_disk_store()._path_for('user').read_bytes() == before
        if approval_delivery_failed:
            assert not selected.get('prompt')
            assert 'approval card could not be delivered' in review.server.calls_of('sendMessage')[-1].params['text']
        else:
            assert selected['prompt']['revision'] == 2
            await click(review, selected, 'approve')
            assert 'Selected exact draft.' in load_on_disk_store().user_entries
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_confirmed_draft_delivery_is_not_repeated(review):
    record, agent, tokens, args = await setup_discussion(review)
    try:
        result = discussion_tool('suggest', content='Frozen candidate.',
            context=source_context(agent), **args)
        first = send_draft(result)
        assert first['success'] and first['delivered']
        receipt = wa.get_pending('memory', record['id'])['discussion_draft']['receipt']
        sends = len(review.server.calls_of('sendMessage'))
        # Replaying registration after transport completed must use its durable receipt.
        assert send_draft(dict(result)) == first
        assert len(review.server.calls_of('sendMessage')) == sends
        assert wa.get_pending('memory', record['id'])['discussion_draft']['receipt'] == receipt
    finally:
        clear_session_vars(tokens)
