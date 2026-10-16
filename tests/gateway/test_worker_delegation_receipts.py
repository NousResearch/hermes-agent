"""Ledger retry durability shares the worker sequence and epoch boundary."""
from contextlib import closing
import os
import sqlite3
import time

import psutil
import pytest

from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from hermes_state_runtime import (RuntimeStoreError, begin_runtime_epoch, register_worker_execution,
                                  adopt_worker_execution, mutate_worker_execution)
from tools.async_delegation import _initialize_schema


def test_lost_ack_reopens_owner_and_replays_same_ledger_receipt(tmp_path):
    from gateway.session_worker_delegation import persist_worker_delegation
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    db.create_session('parent', source='cli')
    epoch = begin_runtime_epoch(db, instance_id='first')
    with closing(sqlite3.connect(path)) as conn:
        _initialize_schema(conn)
    register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent',
                              generation=0, kind='compute', adoption_secret='private')
    scope = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                 pid=os.getpid(), birth=psutil.Process().create_time())
    lost = [True]
    def rpc(method, **params):
        assert method == 'worker.persist'
        pid, birth = params.pop('pid'), params.pop('birth')
        result = persist_worker_delegation(db, **params, worker_pid=pid, worker_birth=birth)
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
        db = SessionDB(path)
        replacement = begin_runtime_epoch(db, instance_id='second')
        with pytest.raises(RuntimeStoreError, match='stale_epoch'):
            store.retry_pending()
        adopted = adopt_worker_execution(db, epoch=replacement, execution_id='worker', session_id='parent',
                                          generation=0, adoption_secret='private')
        store.adopt(adopted['owner_epoch'])
        assert store.retry_pending() == [{'value': None}]
        # Transcript operations and ledger operations cannot consume the same sequence twice.
        mutate_worker_execution(db, epoch=replacement, execution_id='worker', session_id='parent', generation=0,
                                sequence=2, operation='transcript.append', payload={'messages': [{'role': 'user', 'content': 'one'}]})
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            persist_worker_delegation(db, epoch=replacement, execution_id='worker', session_id='parent', generation=0,
                sequence=2, operation='delegation.read', payload={'delegation_id': 'deleg_owned'},
                worker_pid=os.getpid(), worker_birth=scope['birth'])
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM async_delegations').fetchone()[0] == 1
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 2
    finally:
        store.close()
        db.close()


def test_receipt_fences_wrong_session_generation_and_owner_epoch(tmp_path):
    from gateway.session_worker_delegation import persist_worker_delegation
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('parent', source='cli')
        db.create_session('foreign', source='cli')
        epoch = begin_runtime_epoch(db, instance_id='first')
        register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent',
                                  generation=0, kind='compute', adoption_secret='private')
        scope = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
            sequence=1, operation='delegation.read', payload={'delegation_id': 'none'},
            worker_pid=os.getpid(), worker_birth=psutil.Process().create_time())
        for override, error in [({'session_id': 'foreign'}, 'permission_denied'),
                                ({'generation': 1}, 'stale_generation'),
                                ({'epoch': epoch + 1}, 'stale_epoch'),
                                ({'operation': 'execute_sql'}, 'invalid_params')]:
            with pytest.raises(RuntimeStoreError, match=error):
                persist_worker_delegation(db, **(scope | override))
        with db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 0
    finally:
        db.close()
