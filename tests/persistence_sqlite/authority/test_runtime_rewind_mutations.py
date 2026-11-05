"""Rewind fences the generation and preserves archived history."""
import pytest
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import (
    append_message, append_messages_batch, archive_and_compact, get_messages, message_count, rewind_to_message,
)
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session


def test_rewind_has_existing_semantics_with_atomic_retry(store):
    holder = store.holder()
    create_session(holder, 's', 'test')
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'reply'}, {'role': 'user', 'content': 'again'}])
    target = get_messages(holder, 's')[-1]['id']
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='rewind',
        expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': target})
    receipt, _, _ = mutate_runtime_session(holder, **args)
    assert receipt['rewound_count'] == 1 and receipt['target_message']['content'] == 'again'
    assert [m['content'] for m in get_messages(holder, 's')] == ['first', 'reply']
    assert message_count(holder, 's') == 3
    assert get_session(holder, 's')['runtime_generation'] == 1
    assert mutate_runtime_session(holder, **args)[0] == receipt
    assert get_session(holder, 's')['rewind_count'] == 1


def test_rewind_refuses_queued_work(store):
    holder = store.holder()
    create_session(holder, 's', 'test')
    append_message(holder, 's', role='user', content='keep')
    target = get_messages(holder, 's')[0]['id']
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={'text': 'queued'})
    with pytest.raises(AuthorityRefusal, match='session_busy'):
        mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='s', request_id='rewind',
            expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': target})
    assert get_messages(holder, 's')[0]['content'] == 'keep'


def _compacted_session(holder):
    """Q1..A2 archived by in-place compaction (still listed with include_compacted), then Q3/A3."""
    create_session(holder, 's', 'test')
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'Q1'}, {'role': 'assistant', 'content': 'A1'},
                                        {'role': 'user', 'content': 'Q2'}, {'role': 'assistant', 'content': 'A2'}])
    archive_and_compact(holder, 's', [{'role': 'user', 'content': '[CONTEXT SUMMARY] Q1..A2'},
                                      {'role': 'assistant', 'content': 'ok, continuing'}])
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'Q3'}, {'role': 'assistant', 'content': 'A3'}])
    shown = {m['content']: m['id'] for m in get_messages(holder, 's', include_compacted=True)}
    return shown, begin_runtime_epoch(holder, instance_id='owner')


def _rewind_to(holder, epoch, target, request_id='rewind'):
    return mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='s', request_id=request_id,
        expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': target})[0]


def test_rewind_refuses_a_row_compaction_archived(store):
    holder = store.holder()
    shown, epoch = _compacted_session(holder)
    live = ['[CONTEXT SUMMARY] Q1..A2', 'ok, continuing', 'Q3', 'A3']
    assert [m['content'] for m in get_messages(holder, 's')] == live
    with pytest.raises(AuthorityRefusal, match='invalid_params') as refused:
        _rewind_to(holder, epoch, shown['Q2'])
    assert refused.value.field == 'target_message_id'
    assert [m['content'] for m in get_messages(holder, 's')] == live
    assert get_session(holder, 's')['runtime_generation'] == 0
    # The live turn after the compaction still rewinds normally.
    receipt = _rewind_to(holder, epoch, shown['Q3'], request_id='live')
    assert receipt['rewound_count'] == 2
    assert [m['content'] for m in get_messages(holder, 's')] == live[:2]


@pytest.mark.parametrize('target', ['missing', 'assistant'])
def test_rewind_refuses_a_target_that_is_not_a_live_user_row(store, target):
    holder = store.holder()
    shown, epoch = _compacted_session(holder)
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        _rewind_to(holder, epoch, 99999 if target == 'missing' else shown['A3'])
    assert [m['content'] for m in get_messages(holder, 's')][-2:] == ['Q3', 'A3']


def test_rewind_refuses_a_row_a_compression_archived(store):
    """A compacted row is still shown in history, but it is not live: a rewind to it would deactivate
    every live row after it, the whole conversation since the compression."""
    holder = store.holder()
    create_session(holder, 's', 'test')
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'reply'}])
    archived = get_messages(holder, 's')[0]['id']
    archive_and_compact(holder, 's', [{'role': 'user', 'content': '[CONTEXT COMPACTION] earlier turns'}])
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'live'}, {'role': 'assistant', 'content': 'answer'}])
    live = [m['content'] for m in get_messages(holder, 's')]
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id='s', request_id='rewind',
            expected_revision=0, expected_generation=0, operation='rewind', payload={'target_message_id': archived})
    assert [m['content'] for m in get_messages(holder, 's')] == live


def _refusal(holder, epoch, target, request_id):
    with pytest.raises(AuthorityRefusal, match='invalid_params') as refused:
        _rewind_to(holder, epoch, target, request_id=request_id)
    assert refused.value.field == 'target_message_id'
    return refused.value.detail


def test_only_a_compacted_target_carries_the_compaction_marker(store):
    """Desktop says "summarized when the chat was compressed" only for the ``compacted:`` detail: a turn
    another window already undid, an assistant row and an unknown id are stale for other reasons."""
    holder = store.holder()
    shown, epoch = _compacted_session(holder)
    append_messages_batch(holder, 's', [{'role': 'user', 'content': 'Q4'}, {'role': 'assistant', 'content': 'A4'}])
    q4 = get_messages(holder, 's')[-2]['id']
    rewind_to_message(holder, 's', q4)  # another client's /undo: no compaction involved
    compacted = _refusal(holder, epoch, shown['Q2'], 'compacted')
    assert compacted.startswith('compacted:')
    for name, target in {'assistant': shown['A3'], 'undone': q4, 'archived-assistant': shown['A1'],
                         'missing': 99999}.items():
        detail = _refusal(holder, epoch, target, name)
        assert detail != compacted and not detail.startswith('compacted:'), name


def test_a_turn_a_rotating_compression_left_on_the_parent_carries_the_marker(store):
    holder = store.holder()
    create_session(holder, 'p', 'test')
    append_messages_batch(holder, 'p', [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'reply'}])
    old = get_messages(holder, 'p')[0]['id']
    end_session(holder, 'p', 'compression')
    create_session(holder, 's', 'test', parent_session_id='p')
    append_messages_batch(holder, 's', [{'role': 'user', 'content': '[CONTEXT SUMMARY] old'},
                                        {'role': 'user', 'content': 'live'}, {'role': 'assistant', 'content': 'ok'}])
    create_session(holder, 'elsewhere', 'test')
    foreign = append_message(holder, 'elsewhere', 'user', 'not this chat')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    assert _refusal(holder, epoch, old, 'parent').startswith('compacted:')
    assert not _refusal(holder, epoch, foreign, 'foreign').startswith('compacted:')
