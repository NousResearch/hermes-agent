"""A retained worker outbox follows verified owner adoption (epoch replacement, physical-target
rotation) without a second receipt, and is still refused for anything but the verified successor."""
import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError
from hermes_state import SessionDB
from hermes_state_runtime import (adopt_worker_execution, begin_runtime_epoch,
                                  mutate_worker_execution, register_worker_execution)


def _owner(db):
    return lambda method, **params: mutate_worker_execution(db, **params)


def _lose_ack(db, store):
    def lost(method, **params):
        mutate_worker_execution(db, **params)  # commits for real; only the reply is lost
        raise TimeoutError('lost_ack')
    store.rpc = lost


def _receipts(db):
    return db._read_one('SELECT COUNT(*) FROM worker_receipts')[0]


def test_lost_ack_receipt_reconciles_under_the_adopted_epoch_once(tmp_path):
    path, outbox = tmp_path / 'state.db', tmp_path / 'outbox'
    db = SessionDB(path)
    db.create_session('s', 'cli')
    first = begin_runtime_epoch(db, instance_id='first')
    scope = dict(epoch=first, execution_id='worker', session_id='s', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='private')
    store = RuntimeSessionStore(_owner(db), scope, outbox)
    _lose_ack(db, store)
    with pytest.raises(TimeoutError):
        store.queue_token_counts('s', input_tokens=10)
    store._outbox_owner.close()  # the worker exited with its durable pending journal
    db.close()
    db = SessionDB(path)  # owner replaced
    try:
        second = begin_runtime_epoch(db, instance_id='second')
        successor = dict(scope, epoch=second)
        # Not the verified successor: no adoption proof, or proof for another epoch/execution.
        adopted = adopt_worker_execution(db, epoch=second, execution_id='worker', session_id='s',
                                         generation=0, adoption_secret='private')
        for proof in (None, dict(adopted, owner_epoch=first), dict(adopted, execution_id='other')):
            with pytest.raises(WorkerPersistenceError, match='outbox_scope_mismatch'):
                RuntimeSessionStore(_owner(db), successor, outbox, adopted=proof)
        with pytest.raises(WorkerPersistenceError, match='outbox_scope_mismatch'):
            RuntimeSessionStore(_owner(db), dict(successor, generation=1), outbox, adopted=adopted)
        assert _receipts(db) == 1
        reopened = RuntimeSessionStore(_owner(db), successor, outbox, adopted=adopted)
        try:
            assert reopened.journal['pending'] == [] and reopened.scope == successor
            reopened.queue_token_counts('s', input_tokens=5)
            assert _receipts(db) == 2  # sequence 1 resolved as its original receipt, then 2
            assert db.get_session('s')['input_tokens'] == 15
        finally:
            reopened.close()
    finally:
        db.close()


def test_pending_publication_follows_adoption_onto_the_rotated_target(tmp_path):
    path, outbox = tmp_path / 'state.db', tmp_path / 'outbox'
    db = SessionDB(path)
    db.create_session('root', 'cli', system_prompt='PREFIX')
    first = begin_runtime_epoch(db, instance_id='first')
    scope = dict(epoch=first, execution_id='worker', session_id='root', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='private')
    store = RuntimeSessionStore(_owner(db), scope, outbox)
    assert store.try_acquire_compression_lock('root', 'holder')
    _lose_ack(db, store)
    with pytest.raises(TimeoutError):
        store.publish_compression_child(parent_session_id='root', child_session_id='child', source='cli',
            messages=[{'role': 'assistant', 'content': 'SUMMARY'}], system_prompt='PREFIX',
            compression_lock_holder='holder')
    store._outbox_owner.close()
    db.close()
    db = SessionDB(path)
    try:
        second = begin_runtime_epoch(db, instance_id='second')
        adopted = adopt_worker_execution(db, epoch=second, execution_id='worker', session_id='child',
                                         generation=0, adoption_secret='private')
        receipts = _receipts(db)
        reopened = RuntimeSessionStore(_owner(db), dict(scope, epoch=second, session_id='child'),
                                       outbox, adopted=adopted)
        try:
            assert reopened.journal['pending'] == []
            assert reopened.scope == dict(scope, epoch=second, session_id='child')
            assert _receipts(db) == receipts
            assert db._read_one('SELECT COUNT(*) FROM sessions WHERE id=?', ('child',))[0] == 1
            reopened.append_messages_batch('child', [{'role': 'user', 'content': 'next'}])
        finally:
            reopened.close()
    finally:
        db.close()
