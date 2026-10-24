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

# Real loopback WorkerRPC + disposable SQLite owner, with synthetic credentials.
from tests.gateway.test_worker_receipt_reconciliation import owner


def _delegation_params(owner):
    with closing(sqlite3.connect(owner.db.db_path)) as conn:
        _initialize_schema(conn)
    return dict(owner.scope, sequence=1, operation='delegation.dispatch', payload=dict(
        delegation_id='deleg_synthetic', task={'goal': 'inert'}, dispatched_at=time.time(),
        parent_session_id='owned', session_key='', origin_ui_session_id='', origin_session_id=''))


def _delegation_rows(owner):
    with closing(sqlite3.connect(owner.db.db_path)) as conn:
        return (conn.execute('SELECT delegation_id,state FROM async_delegations').fetchall(),
                conn.execute('SELECT sequence,payload_digest,result_json FROM worker_receipts').fetchall())


@pytest.mark.parametrize('flag', ['omitted', False, None])
def test_canonical_delegation_optional_flag_and_immutable_lookup(owner, flag):
    import asyncio
    from gateway.session_controls import AuthorityConnection
    connection = object.__new__(AuthorityConnection)
    connection.authority, connection.actor = owner.authority, owner.connection.actor
    connection._operator = False
    owner.authority.profile_name = None
    params = _delegation_params(owner)
    if flag != 'omitted':
        params['receipt_only'] = flag

    def dispatch(arguments, request_id):
        return asyncio.run_coroutine_threadsafe(connection.dispatch(dict(
            jsonrpc='2.0', id=request_id, method='worker.persist', params=arguments)),
            owner.loop).result(5)

    response = dispatch(params, 1)
    assert response.get('result') == {'value': None}, response
    before = _delegation_rows(owner)
    response = dispatch(dict(params, receipt_only=True), 2)
    assert response.get('result') == {'value': None}, response
    assert _delegation_rows(owner) == before
    assert len(before[0]) == len(before[1]) == 1


def test_native_delegation_lost_ack_recovers_without_mutation_replay(owner):
    import json
    params = _delegation_params(owner)
    owner.mode = 'drop'
    result, error = None, None
    try:
        result = owner.store._apply(params['operation'], params['payload'])
    except Exception as exc:
        error = exc
    rows, receipts = _delegation_rows(owner)
    assert rows == [('deleg_synthetic', 'running')]
    assert len(receipts) == 1 and receipts[0][0] == 1
    assert len(owner.requests) == 2
    assert owner.requests[1] == dict(owner.requests[0], receipt_only=True)
    assert error is None, f'committed delegation receipt not recovered: {error!r}'
    assert result == json.loads(receipts[0][2]) == {'value': None}
    assert owner.store.journal['pending'] == [] and owner.store.failure is None
    assert owner.store.journal['next_sequence'] == 2


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'unknown_receipt'), ('digest', 'admission_conflict'),
    ('epoch', 'stale_epoch'), ('generation', 'stale_generation'),
    ('terminal', 'stale_generation'), ('unknown', 'stale_generation'),
    ('retired', 'runtime_retired'), ('secret', 'permission_denied'),
    ('profile', 'profile_mismatch'), ('capability', 'permission_denied'),
    ('pid', 'invalid_params'), ('birth', 'permission_denied'),
    ('extra_key', 'invalid_params'),
])
def test_delegation_receipt_lookup_preserves_auth_and_unknown_fences(owner, mutation, reason):
    import os
    from agent.runtime_session_store import WorkerPersistenceError
    import hermes_state_runtime as rt
    params = _delegation_params(owner)
    if mutation != 'missing':
        assert owner.rpc('worker.persist', **params) == {'value': None}
    if mutation == 'digest':
        params['payload']['task']['goal'] = 'changed'
    elif mutation == 'epoch':
        rt.begin_runtime_epoch(owner.db, instance_id='replacement')
    elif mutation == 'generation':
        owner.db._execute_write(lambda conn: conn.execute('UPDATE sessions SET runtime_generation=1'))
    elif mutation in ('terminal', 'unknown'):
        owner.db._execute_write(lambda conn: conn.execute('UPDATE worker_executions SET status=?', (mutation,)))
    elif mutation == 'retired':
        owner.authority.retired = True
    elif mutation == 'secret':
        params['secret'] = 'foreign-synthetic-secret'
    elif mutation == 'profile':
        params['profile_id'] = 'foreign-profile'
    elif mutation == 'capability':
        owner.connection.actor.capabilities = {'session:read'}
    elif mutation == 'pid':
        params['pid'] = os.getpid()
    elif mutation == 'birth':
        params['birth'] += 1
    elif mutation == 'extra_key':
        params['unrecognized'] = True
    before = _delegation_rows(owner)
    with pytest.raises(WorkerPersistenceError, match=reason):
        owner.rpc.reconcile_receipt(**params)
    assert _delegation_rows(owner) == before


@pytest.mark.parametrize('flag', [0, 1, 'true', [], {}], ids=['int0', 'int1', 'string', 'list', 'dict'])
def test_delegation_rejects_non_boolean_receipt_flags_without_effect(owner, flag):
    from agent.runtime_session_store import WorkerPersistenceError
    params = _delegation_params(owner)
    with pytest.raises(WorkerPersistenceError, match='invalid_params'):
        owner.rpc('worker.persist', **dict(params, receipt_only=flag))
    assert _delegation_rows(owner) == ([], [])


def test_reopened_delegation_outbox_uses_immutable_receipt_only(owner):
    import json
    params = _delegation_params(owner)
    owner.mode = 'drop'
    def unavailable_lookup(arguments):
        if arguments.get('receipt_only'):
            raise OSError('synthetic receipt temporarily unavailable')
    owner.before_request = unavailable_lookup
    with pytest.raises(Exception):
        owner.store._apply(params['operation'], params['payload'])
    path = owner.store.path
    journal = json.loads(path.read_text())
    assert len(journal['pending']) == 1 and owner.store.failure
    assert len(_delegation_rows(owner)[1]) == 1
    # Simulate a fresh local facade, preserving the original UNKNOWN journal.
    owner.store._outbox_owner.close()
    owner.before_request = lambda arguments: None
    owner.store = RuntimeSessionStore(owner.rpc, owner.scope, path.parent)
    # The constructor itself consumes retained receipts; no second observation is needed.
    assert owner.store.path == path
    assert owner.store.retry_pending() == []
    assert owner.store.journal['pending'] == [] and owner.store.failure is None
    assert owner.store.journal['next_sequence'] == 2
    assert len(owner.requests) == 3
    assert all(row == dict(owner.requests[0], receipt_only=True) for row in owner.requests[1:])
    assert len(_delegation_rows(owner)[0]) == len(_delegation_rows(owner)[1]) == 1


@pytest.mark.parametrize('held_stage', ['checkpoint', 'publication'])
def test_cancelled_delegation_completion_tracks_physical_write_and_publication(owner, monkeypatch, held_stage):
    import asyncio
    import queue
    import threading
    from gateway.session_worker import worker_request
    from gateway.session_runtime_workers import mutation_tasks
    from tools.process_registry import process_registry
    from tests.waiting import async_wait_until

    dispatch = _delegation_params(owner)
    assert owner.store._apply(dispatch['operation'], dispatch['payload']) == {'value': None}
    entered, release, published = threading.Event(), threading.Event(), threading.Event()

    class ObservedQueue(queue.Queue):
        def put(self, value, *args, **kwargs):
            if held_stage == 'publication':
                entered.set()
                assert release.wait(10), 'fixture publication not released'
            super().put(value, *args, **kwargs)
            published.set()

    completion_queue = ObservedQueue()
    monkeypatch.setattr(process_registry, 'completion_queue', completion_queue)
    if held_stage == 'checkpoint':
        def checkpoint():
            entered.set()
            assert release.wait(10), 'fixture checkpoint not released'
        monkeypatch.setattr(owner.db, '_try_wal_checkpoint', checkpoint)
        owner.db._write_count = owner.db._CHECKPOINT_EVERY_N_WRITES - 1

    event = dict(type='async_delegation', delegation_id='deleg_synthetic', status='completed',
                 completed_at=time.time(), parent_session_id='owned', session_key='',
                 origin_ui_session_id='', origin_session_id='')
    params = dict(owner.scope, sequence=2, operation='delegation.complete',
                  payload={'event': event, 'result': {'summary': 'inert result'}})

    async def scenario():
        task = asyncio.create_task(worker_request(owner.connection, owner.ref, params, operation='persist'))
        try:
            await async_wait_until(entered.is_set, timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            rows, receipts = _delegation_rows(owner)
            assert rows == [('deleg_synthetic', 'completed')] and len(receipts) == 2
            assert not published.is_set() and completion_queue.empty()
            assert len(mutation_tasks(owner.authority)) == 1, 'physical delegation must stay retirement-tracked'
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            # Physical observation is independent of the tracker under test.
            await async_wait_until(published.is_set, timeout=5)
            await async_wait_until(lambda: not mutation_tasks(owner.authority), timeout=5)

    asyncio.run_coroutine_threadsafe(scenario(), owner.loop).result(20)
    assert completion_queue.qsize() == 1
    assert completion_queue.get_nowait() == event | {'restored': True}
    before = _delegation_rows(owner)
    assert owner.rpc.reconcile_receipt(**params) == {'value': None}
    assert completion_queue.empty() and _delegation_rows(owner) == before
    with owner.db._read_ctx() as conn:
        row = conn.execute('SELECT delivery_state,delivery_attempts,delivery_claim FROM async_delegations').fetchone()
        assert tuple(row) == ('pending', 0, None), 'publication must not ACK or claim delivery'


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
