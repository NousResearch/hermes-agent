"""Physical worker rotation cannot re-key an accepted input or unblock its FIFO."""
import pytest

from agent.runtime_session_store import RuntimeSessionStore
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input, get_session_admission
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.workers._executions import register_worker_execution
from persistence.sqlite.workers._receipt import mutate_worker_execution


def test_rotation_preserves_admission_identity_and_lost_ack_fifo(backend, scope_a, tmp_path):
    holder = backend.open(scope_a).holder()
    create_session(holder, 'root', 'cli', system_prompt='PREFIX')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    request = dict(epoch=epoch, principal_id='human', session_id='root',
                   request_id='original', payload={'text': 'FIRST'})
    accepted, _ = admit_session_input(holder, **request)
    started = claim_session_input(holder, epoch=epoch, session_id='root')
    follower, _ = admit_session_input(holder, **dict(request, request_id='follower', payload={'text': 'SECOND'}))
    scope = dict(epoch=epoch, execution_id='admission-worker:' + accepted['admission_id'],
                 session_id='root', generation=started['generation'])
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='private')
    rpc = lambda method, **params: mutate_worker_execution(holder, **params)
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
            assert admit_session_input(holder, **request)[0]['admission_id'] == accepted['admission_id']
            assert claim_session_input(holder, epoch=epoch, session_id='root') is None
            store.rpc = rpc
            store.retry_pending()
            assert store.scope['session_id'] == child
            assert holder._read_one('SELECT COUNT(*) FROM sessions WHERE id=?', (child,))[0] == 1
            assert get_session_admission(holder, admission_id=follower['admission_id'])['status'] == 'queued'
        row = get_session_admission(holder, admission_id=accepted['admission_id'])
        assert row['target_session_id'] == 'root'
        assert holder._read_one('SELECT lineage_json FROM session_admissions WHERE admission_id=?',
                                (accepted['admission_id'],))[0] == '["root","child","grandchild"]'
        assert holder._read_one('SELECT COUNT(*) FROM session_admissions')[0] == 2
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


def test_fifo_stays_blocked_while_its_worker_lives_on_a_compression_child(backend, scope_a, tmp_path):
    """Compression moves a live worker onto the physical child while the FIFO stays keyed on the
    logical root. The root's queued input must not be claimed until that worker ends, or two
    writers run in one lineage."""
    holder = backend.open(scope_a).holder()
    create_session(holder, 'root', 'cli', system_prompt='PREFIX')
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    scope = dict(epoch=epoch, execution_id='worker', session_id='root', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='private', require_idle=True)
    store = RuntimeSessionStore(lambda method, **params: mutate_worker_execution(holder, **params),
                                scope, tmp_path / 'outbox')
    try:
        messages = [{'role': 'user', 'content': 'kept'}]
        store.append_messages_batch('root', messages)
        assert store.try_acquire_compression_lock('root', 'holder')
        store.publish_compression_child(parent_session_id='root', child_session_id='child', source='cli',
                                        messages=messages, system_prompt='PREFIX', compression_lock_holder='holder')
        assert store.scope['session_id'] == 'child'
        follower, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='root',
                                          request_id='follower', payload={'text': 'next'})
        assert claim_session_input(holder, epoch=epoch, session_id='root') is None
        store.finish()
        claimed = claim_session_input(holder, epoch=epoch, session_id='root')
        assert claimed is not None and claimed['admission_id'] == follower['admission_id']
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()
