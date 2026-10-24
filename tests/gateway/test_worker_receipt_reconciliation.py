"""Real SQLite commits plus the production synchronous store/RPC, never a model/tool replay."""
import asyncio
import contextlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerRPC
from hermes_state import SessionDB
import hermes_state_runtime as rt
from tests.waiting import wait_until


@pytest.fixture
def owner(tmp_path, monkeypatch):
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery
    from gateway.session_worker import _claim, worker_request

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'cli')
    epoch = rt.begin_runtime_epoch(db, instance_id='synthetic-owner')
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scope = dict(profile_id=str(tmp_path), session_id='owned', execution_id='worker', generation=0,
                 pid=child.pid, birth=psutil.Process(child.pid).create_time(), secret='synthetic-secret', epoch=epoch)
    authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path),
                                _require_admission_open=lambda: None, wake_after_worker=lambda sid: None)
    connection = SimpleNamespace(authority=authority,
        actor=SimpleNamespace(profile_id=str(tmp_path), capabilities={'worker:adopt'}, subject='synthetic'))
    ref = SimpleNamespace(session_id='owned')
    rt.register_worker_execution(db, **{k: scope[k] for k in ('epoch', 'execution_id', 'session_id', 'generation')},
                                 kind='compute', adoption_secret=_claim(connection, ref, scope))
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, name='fixture-owner-loop')
    loop_thread.start()
    state = SimpleNamespace(db=db, scope=scope, authority=authority, requests=[], links=[], handlers=[],
                            entered=threading.Event(), release=threading.Event(), returned=threading.Event(),
                            mode='timely', errors=[], events=[], after_commit=lambda params: None,
                            before_request=lambda params: None)

    def handler(ws):
        state.handlers.append(threading.current_thread())
        try:
            for raw in ws:
                request = json.loads(raw)
                params = request['params']
                state.requests.append(params)
                state.events.append(('request', time.monotonic(), bool(params.get('receipt_only'))))
                try:
                    state.before_request(params)
                    result = asyncio.run_coroutine_threadsafe(
                        worker_request(connection, ref, params, operation='persist'), loop).result(65)
                    state.events.append(('handler_return', time.monotonic(), bool(params.get('receipt_only'))))
                    response = {'id': request['id'], 'result': result}
                except Exception as exc:
                    response = {'id': request['id'], 'error': {'message': str(exc)}}
                if not params.get('receipt_only'):
                    state.after_commit(params)
                    state.returned.set()
                    if state.mode == 'drop':
                        return
                ws.send(json.dumps(response))
        except Exception as exc:
            # A timed-out client deliberately closes its first connection.
            state.errors.append(type(exc).__name__)

    with serve(handler, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1'], close_timeout=1) as server:
        server_thread = threading.Thread(target=server.serve_forever, name='fixture-websocket-server')
        server_thread.start()
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}', control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda *a, **k: 'synthetic-ticket')
        rpc = WorkerRPC(tmp_path)
        real_connection = rpc._connection
        def connect():
            link = real_connection()
            if link not in state.links:
                state.links.append(link)
            return link
        monkeypatch.setattr(rpc, '_connection', connect)
        state.rpc = rpc
        state.connection, state.ref, state.loop = connection, ref, loop
        state.store = RuntimeSessionStore(rpc, scope, tmp_path / 'outbox')
        try:
            yield state
        finally:
            state.release.set()
            rpc.close()
            server.shutdown()
            server_thread.join(5)
            for thread in state.handlers:
                thread.join(5)
            for link in state.links:
                link.reader.join(5)
            # Fixture cleanup never replays or clears a pending journal.
            state.store._outbox_owner.close()
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(5)
            loop.close()
            child.stdin.close()
            child.wait(timeout=5)
            db.close()
            assert not server_thread.is_alive() and not loop_thread.is_alive()
            assert all(not thread.is_alive() for thread in state.handlers)
            assert all(not link.reader.is_alive() for link in state.links)


def committed(owner):
    with contextlib.closing(sqlite3.connect(owner.db.db_path)) as conn:
        return conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0], conn.execute(
            'SELECT sequence,payload_digest,result_json FROM worker_receipts').fetchall()


def test_postcommit_maintenance_beyond_receipt_budget_reconciles_without_replay(owner, monkeypatch):
    """Hold post-COMMIT maintenance past a fixture-bound legacy 20s receive."""
    # The production policy now permits a longer mutation wait. Keep this
    # lost-ACK discriminator at its historical observation boundary without
    # changing product budgets or the separate >20s default-policy positive.
    from agent.runtime_session_store import _OwnerLink
    real_receive = _OwnerLink.receive
    monkeypatch.setattr(_OwnerLink, 'receive',
                        lambda self, timeout: real_receive(self, min(timeout, 20)))
    def maintenance():
        owner.events.append(('postcommit', time.monotonic(), False))
        owner.entered.set()
        assert owner.release.wait(60), 'fixture maintenance gate not released'
    monkeypatch.setattr(owner.db, '_try_wal_checkpoint', maintenance)
    owner.db._write_count = owner.db._CHECKPOINT_EVERY_N_WRITES - 1
    rows = [{'role': 'assistant', 'content': 'synthetic completed tool result'}]
    error = None
    start = time.monotonic()
    try:
        count = owner.store.append_messages_batch('owned', rows)
    except Exception as exc:
        error = exc
        count = None
    elapsed = time.monotonic() - start
    n, receipts = committed(owner)
    assert owner.entered.is_set() and not owner.returned.is_set()
    assert n == 1 and len(receipts) == 1 and receipts[0][0] == 1
    first = owner.requests[0]
    assert receipts[0][1] == rt.admission_fingerprint(canonical_target='owned',
        payload={'operation': first['operation'], 'payload': first['payload']})
    print('POSTCOMMIT', json.dumps({'elapsed': elapsed, 'rows': n, 'receipt_sequence': receipts[0][0],
                                  'digest': receipts[0][1], 'events': owner.events,
                                  'outcome': type(error).__name__ if error else 'acknowledged'}))
    assert error is None, f'committed receipt was not reconciled: {error!r}'
    assert 20 <= elapsed < 45
    assert count == 1 and rows[0]['_row_id'] == json.loads(receipts[0][2])['annotations'][0]['_row_id']
    assert owner.store.journal['pending'] == [] and owner.store.failure is None
    assert len(owner.requests) == 2
    assert owner.requests[1] == dict(first, receipt_only=True)
    from gateway.session_runtime_workers import mutation_tasks
    assert len(mutation_tasks(owner.authority)) == 1, 'physical writer must remain retirement-tracked'
    owner.release.set()
    assert owner.returned.wait(5)
    wait_until(lambda: not mutation_tasks(owner.authority), timeout=5)
    assert committed(owner)[0] == 1


@pytest.mark.parametrize('operation', ['transcript.append', 'session.activity'])
@pytest.mark.parametrize('mode', ['timely', 'drop'])
def test_timely_and_lost_reply_reconcile_exact_receipt(owner, mode, operation):
    owner.mode = mode
    payload = ({'messages': [{'role': 'tool', 'content': 'already executed', 'tool_call_id': 'synthetic'}],
                'turn_lease_holder': None} if operation == 'transcript.append'
               else {'ts': 1234567890.0, 'description': 'synthetic', 'provenance': None})
    result = owner.store._apply(operation, payload)
    count, receipts = committed(owner)
    assert len(receipts) == 1 and receipts[0][0] == 1
    assert json.loads(receipts[0][2]) == result
    assert count == (1 if operation == 'transcript.append' else 0)
    assert owner.store.journal['next_sequence'] == 2 and owner.store.journal['pending'] == []
    assert len(owner.requests) == (1 if mode == 'timely' else 2)
    if mode == 'drop':
        assert owner.requests[1] == dict(owner.requests[0], receipt_only=True)


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'unknown_receipt'), ('digest', 'admission_conflict'),
    ('epoch', 'stale_epoch'), ('generation', 'stale_generation'),
    ('terminal', 'stale_generation'), ('unknown', 'stale_generation'),
    ('retired_authority', 'runtime_retired'), ('foreign_secret', 'permission_denied'),
])
def test_unproven_receipt_never_writes_or_clears_pending(owner, mutation, reason, monkeypatch):
    from agent.runtime_session_store import WorkerPersistenceError
    params = dict(owner.scope, sequence=1, operation='session.activity',
                  payload={'ts': 1234567890.0, 'description': None, 'provenance': None})
    if mutation != 'missing':
        owner.rpc('worker.persist', **params)
    if mutation == 'digest':
        params['payload']['ts'] += 1
    elif mutation == 'epoch':
        rt.begin_runtime_epoch(owner.db, instance_id='new-synthetic-owner')
    elif mutation == 'generation':
        owner.db._execute_write(lambda c: c.execute('UPDATE sessions SET runtime_generation=1'))
    elif mutation in ('terminal', 'unknown'):
        owner.db._execute_write(lambda c: c.execute('UPDATE worker_executions SET status=?', (mutation,)))
    elif mutation == 'retired_authority':
        owner.authority.retired = True
    elif mutation == 'foreign_secret':
        params['secret'] = 'foreign-synthetic-secret'
    before = committed(owner)
    with pytest.raises(WorkerPersistenceError, match=reason):
        owner.rpc.reconcile_receipt(**params)
    assert committed(owner) == before


def test_missing_ack_before_write_leaves_unknown_and_no_replay(owner, monkeypatch):
    from agent.runtime_session_store import WorkerPersistenceError
    from gateway import session_worker
    def precommit_failure(*a, **k):
        raise OSError('synthetic precommit failure')
    monkeypatch.setattr(session_worker, 'mutate_worker_execution', precommit_failure)
    owner.mode = 'drop'
    with pytest.raises(Exception):
        owner.store.touch_session_activity('owned', ts=1234567890.0)
    assert owner.store.failure and len(owner.store.journal['pending']) == 1
    before = len(owner.requests)
    with pytest.raises(WorkerPersistenceError):
        owner.store.append_messages_batch('owned', [{'role': 'assistant', 'content': 'must not run'}])
    assert len(owner.requests) == before == 2 and committed(owner) == (0, [])
    assert json.loads(owner.store.path.read_text())['pending'] == owner.store.journal['pending']


def test_cancelled_observer_keeps_real_writer_tracked_until_settled(owner, monkeypatch):
    from gateway.session_worker import worker_request
    from gateway.session_runtime_workers import mutation_tasks
    from tests.waiting import async_wait_until
    def maintenance():
        owner.entered.set()
        assert owner.release.wait(10)
    monkeypatch.setattr(owner.db, '_try_wal_checkpoint', maintenance)
    owner.db._write_count = owner.db._CHECKPOINT_EVERY_N_WRITES - 1
    params = dict(owner.scope, sequence=1, operation='session.activity',
                  payload={'ts': 1234567890.0, 'description': None, 'provenance': None})
    async def scenario():
        task = asyncio.create_task(worker_request(owner.connection, owner.ref, params, operation='persist'))
        try:
            await async_wait_until(owner.entered.is_set, timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(mutation_tasks(owner.authority)) == 1
            assert len(committed(owner)[1]) == 1
        finally:
            owner.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await async_wait_until(lambda: not mutation_tasks(owner.authority), timeout=5)
    asyncio.run_coroutine_threadsafe(scenario(), owner.loop).result(15)
    assert len(committed(owner)[1]) == 1


def test_receipt_only_field_is_admitted_by_canonical_contract(owner):
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems
    params = dict(owner.scope, sequence=1, operation='session.activity', payload={}, receipt_only=True)
    assert canonical_param_problems(CANONICAL_METHODS['worker.persist'], params) == []


@pytest.mark.parametrize('mutation', ['digest', 'epoch', 'generation', 'terminal', 'unknown'])
def test_lost_reply_with_invalidated_proof_keeps_original_journal(owner, mutation):
    from agent.runtime_session_store import WorkerPersistenceError
    owner.mode = 'drop'
    def invalidate(params):
        if mutation == 'digest':
            owner.db._execute_write(lambda c: c.execute("UPDATE worker_receipts SET payload_digest='wrong'"))
        elif mutation == 'epoch':
            rt.begin_runtime_epoch(owner.db, instance_id='replacement')
        elif mutation == 'generation':
            owner.db._execute_write(lambda c: c.execute('UPDATE sessions SET runtime_generation=1'))
        else:
            owner.db._execute_write(lambda c: c.execute('UPDATE worker_executions SET status=?', (mutation,)))
    owner.after_commit = invalidate
    with pytest.raises(Exception):
        owner.store.touch_session_activity('owned', ts=1234567890.0)
    disk = json.loads(owner.store.path.read_text())
    assert disk['pending'] == owner.store.journal['pending'] and len(disk['pending']) == 1
    assert disk['pending'][0]['sequence'] == 1 and disk['next_sequence'] == 2
    assert owner.store.failure
    with pytest.raises(WorkerPersistenceError):
        owner.store.flush_token_counts()
    with pytest.raises(WorkerPersistenceError):
        owner.store.touch_session_activity('owned')
    assert len(owner.requests) == 2 and len(committed(owner)[1]) == 1


@pytest.mark.parametrize('blocked', ['sqlite_writer', 'executor', 'send'])
def test_delay_discriminators_keep_effects_and_physical_settlement_explicit(owner, monkeypatch, blocked):
    """Scale only the client observation budget for the three negative schedulers."""
    from agent.runtime_session_store import _OwnerLink
    from gateway.session_runtime_workers import mutation_tasks
    from concurrent.futures import ThreadPoolExecutor
    real_receive = _OwnerLink.receive
    monkeypatch.setattr(_OwnerLink, 'receive', lambda self, timeout: real_receive(self, .3))
    busy = None
    pool = None
    blocker = None
    if blocked == 'sqlite_writer':
        busy = sqlite3.connect(owner.db.db_path)
        busy.execute('BEGIN IMMEDIATE')
    elif blocked == 'executor':
        pool = ThreadPoolExecutor(max_workers=1)
        owner.loop.call_soon_threadsafe(owner.loop.set_default_executor, pool)
        def occupy():
            owner.entered.set()
            assert owner.release.wait(10)
        blocker = asyncio.run_coroutine_threadsafe(asyncio.to_thread(occupy), owner.loop)
        assert owner.entered.wait(5)
    else:
        def hold_send(params):
            owner.entered.set()
            assert owner.release.wait(10)
        owner.after_commit = hold_send
    error = None
    try:
        try:
            owner.store.touch_session_activity('owned', ts=1234567890.0)
        except Exception as exc:
            error = exc
        if blocked == 'send':
            assert error is None and owner.store.journal['pending'] == []
            assert len(committed(owner)[1]) == 1
        else:
            assert error is not None and owner.store.failure
            assert len(owner.store.journal['pending']) == 1 and committed(owner) == (0, [])
            assert mutation_tasks(owner.authority), 'observer timeout is not physical cancellation'
        assert len(owner.requests) == 2
        assert owner.requests[1] == dict(owner.requests[0], receipt_only=True)
    finally:
        if busy is not None:
            busy.rollback(); busy.close()
        owner.release.set()
        if blocker is not None:
            blocker.result(5)
        wait_until(lambda: owner.returned.is_set() and not mutation_tasks(owner.authority), timeout=10)
        if pool is not None:
            pool.shutdown(wait=True)
    assert len(committed(owner)[1]) == 1
    if blocked != 'send':
        # A later commit must not silently clear a failed operation already returned to the agent.
        assert owner.store.failure and len(owner.store.journal['pending']) == 1


def test_receipt_only_traverses_actual_canonical_dispatch(owner):
    from gateway.session_controls import AuthorityConnection
    owner.rpc('worker.persist', **dict(owner.scope, sequence=1, operation='session.activity',
              payload={'ts': 1234567890.0, 'description': None, 'provenance': None}))
    connection = object.__new__(AuthorityConnection)
    connection.authority, connection.actor = owner.authority, owner.connection.actor
    connection._operator = False
    owner.authority.profile_name = None
    params = dict(owner.requests[0], receipt_only=True)
    response = asyncio.run_coroutine_threadsafe(connection.dispatch(
        {'jsonrpc': '2.0', 'id': 7, 'method': 'worker.persist', 'params': params}), owner.loop).result(10)
    assert response == {'jsonrpc': '2.0', 'id': 7, 'result': {'value': None}}
    assert len(committed(owner)[1]) == 1


def test_lost_activity_reply_does_not_poison_following_transcript(owner):
    owner.db._execute_write(lambda c: c.execute('UPDATE sessions SET last_activity_at=NULL'))
    owner.mode = 'drop'
    owner.store.touch_session_activity('owned', ts=1234567890.0, description='completed operation')
    with contextlib.closing(sqlite3.connect(owner.db.db_path)) as conn:
        assert conn.execute('SELECT last_activity_at FROM sessions').fetchone()[0] == 1234567890.0
    rows = [{'role': 'assistant', 'content': 'next ordinary append'}]
    assert owner.store.append_messages_batch('owned', rows) == 1
    assert rows[0]['_row_id']
    assert owner.store.journal['next_sequence'] == 3
    assert owner.store.journal['pending'] == [] and owner.store.failure is None
    assert [p['sequence'] for p in owner.requests] == [1, 1, 2, 2]
    assert committed(owner)[0] == 1
