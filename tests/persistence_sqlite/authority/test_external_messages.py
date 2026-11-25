"""Speech receipts fence the owner and count only committed new user messages for review."""
import pytest

from persistence.authority.external_messages import ExternalMessage
from persistence.conversation.messages import TranscriptRead
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_seed import seed_session


def test_replay_and_rollback_never_count_again_for_memory_review(backend, scope_a):
    store = backend.open(scope_a)
    seed_session(store, 'speech', source='gui')
    epoch = store.runtime.begin_epoch(instance_id='owner')

    def record(*messages):
        return store.runtime.append_external_messages(epoch=epoch, session_id='speech',
                                                       messages=messages, memory_interval=2)

    first = ExternalMessage('user', 'First', 'first')
    second = ExternalMessage('user', 'Second', 'second')
    first_receipt = record(first)
    assert not first_receipt.review_due
    assert not record(ExternalMessage('assistant', 'Reply', 'reply')).review_due
    crossed = record(second)
    assert crossed.review_due
    retry = record(first, second)
    assert retry.messages == first_receipt.messages + crossed.messages
    assert not retry.inserted_ids and not retry.review_due
    with pytest.raises(AuthorityRefusal, match='client_id_conflict'):
        record(ExternalMessage('user', 'Rolled back', 'third'), ExternalMessage('user', 'Changed', 'first'))
    third = record(ExternalMessage('user', 'Third', 'third'))
    assert not third.review_due
    assert record(ExternalMessage('user', 'Fourth', 'fourth')).review_due
    assert [row.content for row in store.messages.read_transcript('speech', TranscriptRead())] == [
        'First', 'Reply', 'Second', 'Third', 'Fourth']


def test_stale_owner_cannot_append_or_replay_and_profiles_are_isolated(backend, scope_a, scope_b):
    first, second = backend.open(scope_a), backend.open(scope_b)
    for store in (first, second):
        seed_session(store, 'speech', source='gui')
    epoch = first.runtime.begin_epoch(instance_id='old-owner')
    messages = (ExternalMessage('user', 'Private speech', 'one'),)
    receipt = first.runtime.append_external_messages(epoch=epoch, session_id='speech', messages=messages)
    current = first.runtime.begin_epoch(instance_id='new-owner')
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        first.runtime.append_external_messages(epoch=epoch, session_id='speech', messages=messages)
    replay = first.runtime.append_external_messages(epoch=current, session_id='speech', messages=messages)
    assert replay.messages == receipt.messages and not replay.inserted_ids
    assert not second.messages.read_transcript('speech', TranscriptRead())
