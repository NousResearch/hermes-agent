"""Physical worker rotation cannot re-key an accepted input or unblock its FIFO."""
import pytest

from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_runtime import (
    admit_session_input, begin_runtime_epoch, claim_session_input,
    get_session_admission, mutate_worker_execution, register_worker_execution,
)


def test_rotation_preserves_admission_identity_and_lost_ack_fifo(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('root', 'cli', system_prompt='PREFIX')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    request = dict(epoch=epoch, principal_id='human', session_id='root',
                   request_id='original', payload={'text': 'FIRST'})
    accepted = admit_session_input(db, **request)
    started = claim_session_input(db, epoch=epoch, session_id='root')
    follower = admit_session_input(db, **dict(request, request_id='follower', payload={'text': 'SECOND'}))
    scope = dict(epoch=epoch, execution_id='admission-worker:' + accepted['admission_id'],
                 session_id='root', generation=started['generation'])
    register_worker_execution(db, **scope, kind='compute', adoption_secret='private')
    rpc = lambda method, **params: mutate_worker_execution(db, **params)
    store = RuntimeSessionStore(rpc, scope, tmp_path / 'outbox')
    try:
        for parent, child in [('root', 'child'), ('child', 'grandchild')]:
            assert store.try_acquire_compression_lock(parent, 'holder')

            def lost_ack(method, **params):
                rpc(method, **params)
                raise TimeoutError('lost publication reply')

            store.rpc = lost_ack
            with pytest.raises(TimeoutError, match='lost publication reply'):
                store.publish_compression_child(parent_session_id=parent, child_session_id=child,
                    source='cli', messages=[{'role': 'assistant', 'content': 'SUMMARY'}],
                    system_prompt='PREFIX', compression_lock_holder='holder')
            assert store.scope['session_id'] == parent
            assert admit_session_input(db, **request)['admission_id'] == accepted['admission_id']
            assert claim_session_input(db, epoch=epoch, session_id='root') is None
            store.rpc = rpc
            store.retry_pending()
            assert store.scope['session_id'] == child
            assert db._read_one('SELECT COUNT(*) FROM sessions WHERE id=?', (child,))[0] == 1
            assert get_session_admission(db, admission_id=follower['admission_id'])['status'] == 'queued'
        row = get_session_admission(db, admission_id=accepted['admission_id'])
        assert row['target_session_id'] == 'root'
        assert db._read_one('SELECT lineage_json FROM session_admissions WHERE admission_id=?',
                            (accepted['admission_id'],))[0] == '["root","child","grandchild"]'
        assert db._read_one('SELECT COUNT(*) FROM session_admissions')[0] == 2
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()
        db.close()
