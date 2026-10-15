"""Same-turn carrier tokens stay owned across worker receipts and retries."""
from contextlib import contextmanager
from types import SimpleNamespace
import json
import os
import time

import psutil
import pytest

from agent.runtime_session_store import RuntimeSessionStore
from gateway.session_worker_delegation import persist_worker_delegation, worker_delegation_handlers
from hermes_state import SessionDB
from hermes_state_errors import DelegationClaimLostError
from hermes_state_runtime import (RuntimeStoreError, begin_runtime_epoch, mutate_worker_execution,
                                  register_worker_execution)
from tools.async_delegation_worker import WorkerDelegationLedger


@contextmanager
def _worker(tmp_path, monkeypatch, *, delivery="inject", complete=True):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    epoch = begin_runtime_epoch(db, instance_id='owner')
    register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='parent',
                              generation=0, kind='compute', adoption_secret='private')
    scope = dict(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                 pid=os.getpid(), birth=1.0)
    faults = {}

    def rpc(method, **params):
        assert method == 'worker.persist'
        pid, birth = params.pop('pid'), params.pop('birth')
        if params['operation'].startswith('delegation.'):
            return persist_worker_delegation(db, **params, worker_pid=pid, worker_birth=birth)
        hook = faults.pop('before_append', None)
        if hook:
            hook()
        result = mutate_worker_execution(db, **params)
        if faults.pop('lose_append_ack', False):
            raise TimeoutError('lost_append_ack')
        return result

    store = RuntimeSessionStore(rpc, scope, tmp_path / 'outbox')
    remote = WorkerDelegationLedger(store)
    record = dict(delegation_id='deleg_owned', parent_session_id='parent', goal='inert',
                  dispatched_at=time.time(), session_key='', origin_ui_session_id='', origin_session_id='',
                  result_delivery=delivery, parent_turn_id='turn-current' if delivery == 'inject' else '')
    # These tests own transaction/receipt semantics; process authentication is
    # covered by the existing worker bridge suite, not an OS subprocess here.
    with monkeypatch.context() as identity:
        identity.setattr(psutil, 'Process', lambda _pid=None: SimpleNamespace(create_time=lambda: 1.0))
        remote.dispatch(record)
    event = dict(record, type='async_delegation', status='completed', completed_at=time.time(), summary='EVIDENCE')
    if complete:
        remote.complete(event, {'summary': 'EVIDENCE'})
        assert remote.claim('deleg_owned', 'carrier-token')
    try:
        yield db, store, remote, faults, scope, event
    finally:
        store.close()
        db.close()


def _carrier():
    return [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call', 'tool_name': 'terminal', 'content': 'original EVIDENCE',
         'display_metadata': {'delegation_delivery': 'tool_boundary', 'delegation_event_ids': ['deleg_owned']},
         '_delegation_delivery_claims': {'deleg_owned': 'carrier-token'}},
    ]


@pytest.mark.parametrize('compression_append', [False, True])
def test_worker_lost_token_receipt_allows_ordinary_fallback_without_foreign_release(
        tmp_path, monkeypatch, compression_append):
    with _worker(tmp_path, monkeypatch) as (db, store, remote, faults, _scope, _event):
        assert remote.status('deleg_owned', 'carrier-token') == {'state': 'pending', 'owned': True}
        assert remote.renew('deleg_owned', 'carrier-token')
        assert not remote.renew('deleg_owned', 'foreign-token')
        rows = _carrier()

        def takeover():
            db._execute_write(lambda conn: conn.execute(
                "UPDATE async_delegations SET delivery_claim='foreign-token' WHERE delegation_id='deleg_owned'"))

        before_revision = db.get_session('parent')['runtime_revision']
        faults['before_append'] = takeover
        kwargs = {'turn_lease_ttl_seconds': 120.0} if compression_append else {}
        with pytest.raises(DelegationClaimLostError):
            store.append_messages_batch('parent', rows, **kwargs)
        # The owner rejected the carrier in its transaction, but committed that
        # refusal's receipt. A fallback is now a new outbox entry, not a poisoned retry.
        assert store.journal['pending'] == [] and store.failure is None
        assert not db.get_messages('parent')
        assert db.get_session('parent')['runtime_revision'] == before_revision
        assert remote.status('deleg_owned', 'carrier-token') == {'state': 'pending', 'owned': False}
        assert not remote.release('deleg_owned', 'carrier-token')
        rows[-1]['content'] = 'original'
        rows[-1].pop('_delegation_delivery_claims')
        rows[-1].pop('display_metadata')
        assert store.append_messages_batch('parent', rows, **kwargs) == 2
        tool_rows = [row for row in db.get_messages('parent') if row['role'] == 'tool']
        assert [row['content'] for row in tool_rows] == ['original']
        with db._read_ctx() as conn:
            assert conn.execute('SELECT delivery_claim FROM async_delegations').fetchone()[0] == 'foreign-token'
            receipts = [json.loads(row[0]) for row in conn.execute('SELECT result_json FROM worker_receipts')]
        assert sum(result.get('delegation_claim_lost') is True for result in receipts) == 1
        assert 'carrier-token' not in str(db.get_messages('parent'))
        assert remote.ack('deleg_owned', 'foreign-token')


def test_worker_committed_carrier_wins_takeover_and_lost_ack_replays_once(tmp_path, monkeypatch):
    with _worker(tmp_path, monkeypatch) as (db, store, remote, faults, scope, _event):
        foreign = worker_delegation_handlers('foreign-execution', scope['pid'], scope['birth'])
        for name in ('delegation.status', 'delegation.renew'):
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                db._execute_write(lambda conn: foreign[name](
                    db, conn, 'parent', {'delegation_id': 'deleg_owned', 'claim_id': 'carrier-token'}))
        owner = worker_delegation_handlers('worker', scope['pid'], scope['birth'])
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            db._execute_write(lambda conn: owner['delegation.complete'](
                db, conn, 'parent', {'event': dict(_event, parent_turn_id='foreign-turn'), 'result': {}}))
        faults['lose_append_ack'] = True
        rows = _carrier()
        with pytest.raises(TimeoutError, match='lost_append_ack'):
            store.append_messages_batch('parent', rows)
        handlers = worker_delegation_handlers('worker', scope['pid'], scope['birth'])
        contender = db._execute_write(lambda conn: handlers['delegation.claim'](
            db, conn, 'parent', {'delegation_id': 'deleg_owned', 'claim_id': 'contender'}))
        assert contender == {'value': False}
        assert len(store.retry_pending()) == 1
        assert store.journal['pending'] == [] and store.failure is None
        assert remote.status('deleg_owned', 'carrier-token') == {'state': 'delivered', 'owned': False}
        tool_rows = [row for row in db.get_messages('parent') if row['role'] == 'tool']
        assert [row['content'] for row in tool_rows] == ['original EVIDENCE']
        assert 'carrier-token' not in str(db.get_messages('parent'))
        with db._read_ctx() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM worker_receipts WHERE result_json LIKE '%annotations%'"
            ).fetchone()[0] == 1


def test_worker_after_turn_default_keeps_ordinary_delivery(tmp_path, monkeypatch):
    with _worker(tmp_path, monkeypatch, delivery='after_turn') as (db, _store, remote, _faults, scope, event):
        assert remote.status('deleg_owned', 'carrier-token') == {'state': 'pending', 'owned': True}
        assert remote.ack('deleg_owned', 'carrier-token')
        assert remote.status('deleg_owned')['state'] == 'delivered'
        owner = worker_delegation_handlers('worker', scope['pid'], scope['birth'])
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            db._execute_write(lambda conn: owner['delegation.complete'](
                db, conn, 'parent', {'event': dict(event, result_delivery='inject', parent_turn_id='forged'),
                                    'result': {}}))
        assert db.get_messages('parent') == []


@pytest.mark.parametrize('delivery', ['inject', 'after_turn'])
def test_worker_delegation_keeps_immutable_parent_across_verified_compression(tmp_path, monkeypatch, delivery):
    from agent.delegation_inject import attach_ready_injects_to_tool_results, acknowledge_pending_injects
    from agent.session_persistence import _db_flush_row, _db_flush_write
    from tools import async_delegation as ledger
    from tools.process_registry import process_registry
    import queue

    with _worker(tmp_path, monkeypatch, delivery=delivery, complete=False) as (
            db, store, remote, _faults, scope, event):
        for parent, child in [('parent', 'child'), ('child', 'grandchild')]:
            holder = 'compression-' + parent
            assert store.try_acquire_compression_lock(parent, holder)
            store.publish_compression_child(
                parent_session_id=parent, child_session_id=child, source='cli',
                messages=[{'role': 'user', 'content': 'compressed context'}], compression_lock_holder=holder)
        assert store.scope['session_id'] == 'grandchild'
        # The child finishes with the immutable parent/turn captured at dispatch.
        remote.complete(event, {'summary': 'EVIDENCE'})
        assert remote.status('deleg_owned')['state'] == 'pending'
        if delivery == 'after_turn':
            assert remote.claim('deleg_owned', 'ordinary-token')
            assert remote.ack('deleg_owned', 'ordinary-token')
            return

        monkeypatch.setattr(ledger, 'worker_ledger', lambda: remote)
        monkeypatch.setattr(process_registry, 'completion_queue', queue.Queue())
        process_registry.completion_queue.put(event)
        messages = _carrier()
        messages[-1]['content'] = 'original'
        messages[-1].pop('_delegation_delivery_claims')
        messages[-1].pop('display_metadata')
        agent = SimpleNamespace(session_id='grandchild', _active_turn_id='turn-current',
                                _session_db=store, _pending_delegation_inject_claims=[])
        assert attach_ready_injects_to_tool_results(agent, messages, 1) == 1
        heartbeat = agent._delegation_inject_claim_heartbeat
        heartbeat['stop'].set()
        heartbeat['thread'].join(timeout=2)
        rows = [_db_flush_row(agent, message, False) for message in messages]
        _db_flush_write(agent, rows, messages, messages)
        assert messages[-1]['_db_persisted'] is True
        handlers = worker_delegation_handlers('worker', scope['pid'], scope['birth'])
        assert db._execute_write(lambda conn: handlers['delegation.claim'](
            db, conn, 'grandchild', {'delegation_id': 'deleg_owned', 'claim_id': 'contender'})) == {'value': False}
        assert acknowledge_pending_injects(agent) == 1
        assert not agent._pending_delegation_inject_claims
        assert remote.status('deleg_owned')['state'] == 'delivered'
        with db._read_ctx() as conn:
            assert conn.execute('SELECT parent_session_id FROM async_delegations').fetchone()[0] == 'parent'

        # Already committed display metadata survives later compaction without
        # resurrecting transient token admission fields.
        assert store.try_acquire_compression_lock('grandchild', 'last-compression')
        store.publish_compression_child(
            parent_session_id='grandchild', child_session_id='great-grandchild', source='cli',
            messages=messages, compression_lock_holder='last-compression')
        assert remote.status('deleg_owned')['state'] == 'delivered'
        copies = [row for row in db.get_messages('great-grandchild') if row['role'] == 'tool']
        assert len(copies) == 1 and 'EVIDENCE' in copies[0]['content']

        db.create_session('branch', source='cli', parent_session_id='parent',
                          model_config={'_branched_from': 'parent'})
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            db._execute_write(lambda conn: handlers['delegation.status'](
                db, conn, 'branch', {'delegation_id': 'deleg_owned', 'claim_id': ''}))
