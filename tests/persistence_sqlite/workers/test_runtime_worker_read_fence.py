"""Worker read receipts leave the session CAS fence where a client last saw it."""
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.conversation._store.sessions import get_session
from tests.persistence_sqlite.workers._runtime_worker import setup_store


def test_worker_reads_keep_revision_and_writes_still_advance_it(backend, scope_a, tmp_path):
    holder, store, scope = setup_store(backend.open(scope_a), tmp_path)
    try:
        store.append_messages_batch('owned', [{'role': 'user', 'content': 'hello'}])
        seen = get_session(holder, 'owned')['runtime_revision']
        store.get_session('owned')
        store.session_lifecycle_statuses(['owned'])
        store.get_messages_as_conversation('owned')
        store.get_compression_tip('owned')
        assert get_session(holder, 'owned')['runtime_revision'] == seen
        # A client mutation built on the pre-read revision is not refused by those reads.
        result, _, _ = mutate_runtime_session(holder, epoch=scope['epoch'], principal_id='client', session_id='owned',
                                              request_id='rename', expected_revision=seen,
                                              operation='rename', payload={'title': 'renamed'})
        assert result['revision'] == seen + 1
        store.touch_session_activity('owned', description='working')
        assert get_session(holder, 'owned')['runtime_revision'] == seen + 2
        store.finish()
        assert get_session(holder, 'owned')['runtime_revision'] == seen + 3
    finally:
        store.close()
