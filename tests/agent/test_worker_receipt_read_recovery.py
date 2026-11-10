"""A timeout settles from an exact owner receipt, never a second mutation."""
import queue
import pytest
from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError, _OwnerLink


def missing_ack():
    link = _OwnerLink.__new__(_OwnerLink)
    link.replies = queue.SimpleQueue()
    return link.receive(0)


def test_timeout_reconciles_without_resending(tmp_path):
    calls = []
    def rpc(method, **params):
        calls.append((method, params))
        if method == 'worker.persist':
            return missing_ack()
        return {'value': True}
    store = RuntimeSessionStore(rpc, {'session_id': 'assigned'}, tmp_path)
    try:
        assert store.try_acquire_session_turn_lease('assigned', 'holder')
        assert [m for m, _ in calls] == ['worker.persist', 'worker.receipt']
        assert calls[0][1] == calls[1][1]
        assert not store.journal['pending'] and store.failure is None
    finally:
        store.close()


def test_lost_committed_publication_recovers_read_only_and_moves_the_outbox(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, register_worker_execution, mutate_worker_execution
    from hermes_state_worker_receipts import read_worker_receipt
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('root', 'cli', system_prompt='PREFIX')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        scope = dict(epoch=epoch, execution_id='worker', session_id='root', generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='synthetic')
        calls = []
        def rpc(method, **params):
            calls.append(method)
            if method == 'worker.receipt':
                return read_worker_receipt(db, **params)
            result = mutate_worker_execution(db, **params)
            if params['operation'] == 'compression.publish':
                return missing_ack()
            return result
        with_store = RuntimeSessionStore(rpc, scope, tmp_path / 'outbox')
        try:
            assert with_store.try_acquire_compression_lock('root', 'holder')
            with_store.publish_compression_child(parent_session_id='root', child_session_id='child', source='cli',
                messages=[{'role': 'assistant', 'content': 'SUMMARY'}], system_prompt='PREFIX',
                compression_lock_holder='holder')
            assert calls == ['worker.persist', 'worker.persist', 'worker.receipt']
            assert with_store.scope['session_id'] == 'child'
            assert db._read_one('SELECT COUNT(*) FROM worker_receipts')[0] == 2
            assert db._read_one('SELECT COUNT(*) FROM sessions WHERE id=?', ('child',))[0] == 1
            assert not with_store.journal['pending']
        finally:
            with_store.close()


def test_receive_budget_exceeds_owner_write_patience(tmp_path, monkeypatch):
    from agent.runtime_session_store import WorkerRPC
    from hermes_state import SessionDB
    budgets = []
    class Link:
        def send(self, request_id, message):
            pass
        def receive(self, timeout):
            budgets.append(timeout)
            return {'result': {}}
    rpc = WorkerRPC(tmp_path)
    monkeypatch.setattr(rpc, '_connection', lambda: Link())
    rpc('worker.persist')
    rpc('worker.receipt')
    assert budgets[0] > SessionDB._TRANSCRIPT_WRITE_PATIENCE_S
    assert 0 < budgets[1] < budgets[0]


def test_missing_receipt_keeps_exact_pending_entry_and_sticky_failure(tmp_path):
    calls = []
    def rpc(method, **params):
        calls.append(method)
        if method == 'worker.persist':
            return missing_ack()
        raise WorkerPersistenceError('not_found')
    store = RuntimeSessionStore(rpc, {'session_id': 'assigned'}, tmp_path)
    with pytest.raises(WorkerPersistenceError, match='not_found'):
        store.try_acquire_session_turn_lease('assigned', 'holder')
    pending = store.journal['pending'].copy()
    with pytest.raises(WorkerPersistenceError, match='not_found'):
        store.try_acquire_session_turn_lease('assigned', 'holder')
    assert store.journal['pending'] == pending
    assert calls == ['worker.persist', 'worker.receipt']
    with pytest.raises(WorkerPersistenceError):
        store.close()
