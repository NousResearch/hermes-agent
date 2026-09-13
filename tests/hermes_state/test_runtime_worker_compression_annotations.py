"""Compaction receipts must refresh the producer's row identities."""
import pytest

from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution


@pytest.mark.parametrize('rotate', [False, True])
def test_compacted_messages_keep_current_row_ids_across_repeated_compaction(tmp_path, rotate):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('root', 'cli', system_prompt='PREFIX')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    scope = dict(epoch=epoch, execution_id='worker', session_id='root', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
    store = RuntimeSessionStore(lambda method, **params: mutate_worker_execution(db, **params),
                                scope, tmp_path / 'outbox')
    messages = [{'role': 'user', 'content': 'retained current input'}]
    store.append_messages_batch('root', messages)
    try:
        for child in ('child', 'grandchild'):
            parent = store.scope['session_id']
            old_id = messages[0]['_row_id']
            assert store.try_acquire_compression_lock(parent, 'holder')
            if rotate:
                store.publish_compression_child(parent_session_id=parent, child_session_id=child,
                    source='cli', messages=messages, system_prompt='PREFIX', compression_lock_holder='holder')
            else:
                store.archive_and_compact(parent, messages, lock_holder='holder')
            current = store.scope['session_id']
            assert messages[0]['_row_id'] != old_id
            assert db._read_one('SELECT session_id,active FROM messages WHERE id=?',
                                (messages[0]['_row_id'],))[:] == (current, 1)
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()
        db.close()
