"""Ledger retry durability shares the worker sequence and epoch boundary."""
import os
import time

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from agent.runtime_session_store import RuntimeSessionStore
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_lost_ack_reopens_owner_and_replays_same_ledger_receipt(tmp_path):
    from gateway.session_worker_delegation import persist_worker_delegation
    db = storage_handle(tmp_path)
    seed_session(db, 'parent', 'cli')
    handle = storage_handle(tmp_path)
    epoch = db.runtime.begin_epoch(instance_id='first')
    db.worker_executions.register(epoch=epoch, execution_id='worker', session_id='parent',
                                  generation=0, kind='compute', adoption_secret='private')
    scope = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                 pid=os.getpid(), birth=worker_birth(psutil.Process()))
    lost = [True]
    def rpc(method, **params):
        assert method == 'worker.persist'
        pid, birth = params.pop('pid'), params.pop('birth')
        result = persist_worker_delegation(handle, **params, worker_pid=pid, worker_birth=birth)
        if lost.pop() if lost else False:
            raise TimeoutError('lost_ack')
        return result
    store = RuntimeSessionStore(rpc, scope, tmp_path / 'outbox')
    try:
        payload = dict(delegation_id='deleg_owned', task={'goal': 'inert'}, dispatched_at=time.time(),
                       parent_session_id='parent', session_key='', origin_ui_session_id='', origin_session_id='')
        with pytest.raises(TimeoutError, match='lost_ack'):
            store._apply('delegation.dispatch', payload)
        db.close()
        db = storage_handle(tmp_path)
        replacement = db.runtime.begin_epoch(instance_id='second')
        with pytest.raises(AuthorityRefusal, match='stale_epoch'):
            store.retry_pending()
        adopted = db.worker_executions.adopt(epoch=replacement, execution_id='worker', session_id='parent',
                                             generation=0, adoption_secret='private')
        store.adopt(adopted.owner_epoch)
        assert store.retry_pending() == [{'value': None}]
        # Transcript operations and ledger operations cannot consume the same sequence twice.
        db.worker_executions.persist_operation(epoch=replacement, execution_id='worker', session_id='parent', generation=0,
                                               sequence=2, operation='transcript.append', payload={'messages': [{'role': 'user', 'content': 'one'}]})
        with pytest.raises(AuthorityRefusal, match='admission_conflict'):
            persist_worker_delegation(handle, epoch=replacement, execution_id='worker', session_id='parent', generation=0,
                sequence=2, operation='delegation.read', payload={'delegation_id': 'deleg_owned'},
                worker_pid=os.getpid(), worker_birth=scope['birth'])
        with raw_holder(db)._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0] == 1
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 2
    finally:
        store.close()
        db.close()


def test_receipt_fences_wrong_session_generation_and_owner_epoch(tmp_path):
    from gateway.session_worker_delegation import persist_worker_delegation
    db = storage_handle(tmp_path)
    try:
        seed_session(db, 'parent', 'cli')
        seed_session(db, 'foreign', 'cli')
        handle = storage_handle(tmp_path)
        epoch = db.runtime.begin_epoch(instance_id='first')
        db.worker_executions.register(epoch=epoch, execution_id='worker', session_id='parent',
                                      generation=0, kind='compute', adoption_secret='private')
        scope = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
            sequence=1, operation='delegation.read', payload={'delegation_id': 'none'},
            worker_pid=os.getpid(), worker_birth=worker_birth(psutil.Process()))
        for override, error in [({'session_id': 'foreign'}, 'permission_denied'),
                                ({'generation': 1}, 'stale_generation'),
                                ({'epoch': epoch + 1}, 'stale_epoch'),
                                ({'operation': 'execute_sql'}, 'invalid_params')]:
            with pytest.raises(AuthorityRefusal, match=error):
                persist_worker_delegation(handle, **(scope | override))
        with raw_holder(db)._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 0
    finally:
        db.close()
