"""A retained worker outbox follows verified owner adoption (epoch replacement, physical-target
rotation) without a second receipt, and is still refused for anything but the verified successor."""
from dataclasses import asdict

import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


def _owner(db):
    return lambda method, **params: db.worker_executions.persist_operation(**params)


def _lose_ack(db, store):
    def lost(method, **params):
        db.worker_executions.persist_operation(**params)  # commits for real; only the reply is lost
        raise TimeoutError('lost_ack')
    store.rpc = lost


def _count(db, sql, params=()):
    with raw_holder(db)._read_ctx() as conn:
        return conn.execute(sql, params).fetchone()[0]


def _receipts(db):
    return _count(db, 'SELECT COUNT(*) FROM worker_receipts')


def test_lost_ack_receipt_reconciles_under_the_adopted_epoch_once(tmp_path):
    outbox = tmp_path / 'outbox'
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, 's', 'cli')
    first = db.runtime.begin_epoch(instance_id='first')
    scope = dict(epoch=first, execution_id='worker', session_id='s', generation=0)
    db.worker_executions.register(**scope, kind='compute', adoption_secret='private')
    store = RuntimeSessionStore(_owner(db), scope, outbox)
    _lose_ack(db, store)
    with pytest.raises(TimeoutError):
        store.queue_token_counts('s', input_tokens=10)
    store._outbox_owner.close()  # the worker exited with its durable pending journal
    db.close()
    db = open_persistence(profile_scope(home=tmp_path))  # owner replaced
    try:
        second = db.runtime.begin_epoch(instance_id='second')
        successor = dict(scope, epoch=second)
        # Not the verified successor: no adoption proof, or proof for another epoch/execution.
        adopted = asdict(db.worker_executions.adopt(epoch=second, execution_id='worker', session_id='s',
                                                    generation=0, adoption_secret='private'))
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
            assert session_row(db, 's')['input_tokens'] == 15
        finally:
            reopened.close()
    finally:
        db.close()


def test_pending_publication_follows_adoption_onto_the_rotated_target(tmp_path):
    outbox = tmp_path / 'outbox'
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, 'root', 'cli', system_prompt='PREFIX')
    first = db.runtime.begin_epoch(instance_id='first')
    scope = dict(epoch=first, execution_id='worker', session_id='root', generation=0)
    db.worker_executions.register(**scope, kind='compute', adoption_secret='private')
    store = RuntimeSessionStore(_owner(db), scope, outbox)
    assert store.try_acquire_compression_lock('root', 'holder')
    _lose_ack(db, store)
    with pytest.raises(TimeoutError):
        store.publish_compression_child(parent_session_id='root', child_session_id='child', source='cli',
            messages=[{'role': 'assistant', 'content': 'SUMMARY'}], system_prompt='PREFIX',
            compression_lock_holder='holder')
    store._outbox_owner.close()
    db.close()
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        second = db.runtime.begin_epoch(instance_id='second')
        adopted = asdict(db.worker_executions.adopt(epoch=second, execution_id='worker', session_id='child',
                                                    generation=0, adoption_secret='private'))
        receipts = _receipts(db)
        reopened = RuntimeSessionStore(_owner(db), dict(scope, epoch=second, session_id='child'),
                                       outbox, adopted=adopted)
        try:
            assert reopened.journal['pending'] == []
            assert reopened.scope == dict(scope, epoch=second, session_id='child')
            assert _receipts(db) == receipts
            assert _count(db, 'SELECT COUNT(*) FROM sessions WHERE id=?', ('child',)) == 1
            reopened.append_messages_batch('child', [{'role': 'user', 'content': 'next'}])
        finally:
            reopened.close()
    finally:
        db.close()
