"""Compression's pre-publication flush uses the ordinary annotated append helper."""
from tests.state.test_runtime_worker_compression import worker  # noqa: F401


def test_compression_flush_returns_canonical_annotations_and_keeps_turn_guard(worker):
    db, store = worker
    assert store.try_acquire_session_turn_lease('owned', 'turn')
    assert store.try_acquire_compression_lock('owned', 'compression')
    rows = [{'role': 'user', 'content': 'accepted', 'api_content': ' original wire ',
             'display_metadata': {'_accepted_input_id': 'input'}}]
    assert store.append_messages_batch('owned', rows, compression_lock_holder='compression',
        turn_lease_holder='turn', turn_lease_ttl_seconds=42) == 1
    assert rows[0]['_row_id'] == db.get_messages_as_conversation('owned', include_row_ids=True)[0]['_row_id']
    assert db.get_messages_as_conversation('owned')[0]['api_content'] == ' original wire '
    import pytest
    from hermes_state_runtime import mutate_worker_execution
    with pytest.raises(Exception, match='lease'):
        mutate_worker_execution(db, **store.scope, sequence=store.journal['next_sequence'],
            operation='compression.append', payload=dict(messages=[{'role': 'assistant', 'content': 'stale'}],
                compression_lock_holder='compression', turn_lease_holder='stale', turn_lease_ttl_seconds=42))
    assert len(db.get_messages('owned')) == 1
