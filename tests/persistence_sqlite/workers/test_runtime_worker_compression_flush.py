"""Compression's pre-publication flush uses the ordinary annotated append helper."""
from persistence.sqlite.conversation._store.messages import get_messages, get_messages_as_conversation
from tests.persistence_sqlite.workers._runtime_worker import worker as worker  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


def test_compression_flush_returns_canonical_annotations_and_keeps_turn_guard(worker):
    persistence, store = worker
    holder = persistence.holder()
    assert store.try_acquire_session_turn_lease('owned', 'turn')
    assert store.try_acquire_compression_lock('owned', 'compression')
    rows = [{'role': 'user', 'content': 'accepted', 'api_content': ' original wire ',
             'display_metadata': {'_accepted_input_id': 'input'}}]
    assert store.append_messages_batch('owned', rows, compression_lock_holder='compression',
        turn_lease_holder='turn', turn_lease_ttl_seconds=42) == 1
    assert rows[0]['_row_id'] == get_messages_as_conversation(holder, 'owned', include_row_ids=True)[0]['_row_id']
    assert get_messages_as_conversation(holder, 'owned')[0]['api_content'] == ' original wire '
    import pytest
    from persistence.sqlite.workers._receipt import mutate_worker_execution
    with pytest.raises(Exception, match='lease'):
        mutate_worker_execution(holder, **store.scope, sequence=store.journal['next_sequence'],
            operation='compression.append', payload=dict(messages=[{'role': 'assistant', 'content': 'stale'}],
                compression_lock_holder='compression', turn_lease_holder='stale', turn_lease_ttl_seconds=42))
    assert len(get_messages(holder, 'owned')) == 1
