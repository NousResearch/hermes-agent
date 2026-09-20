"""Canonical room controls retain the original attempt fence."""
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


def test_unknown_discard_is_exact_and_never_requeues(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    class Service(CanonicalHostedRoomService):
        pass
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = Service(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db.db_path, identity, payload={'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
        lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1, process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        later = time.time() + 31
        monkeypatch.setattr(service.runtime, 'clock', lambda: later)
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)
        # Only the remote canonical boundary is substituted; the room task/lease is real.
        calls = []
        rpc = SimpleNamespace(ref=SimpleNamespace(session_id='session'),
            info=lambda **kw: {'status': 'unknown'},
            discard=lambda **kw: calls.append(kw) or {
                'discarded': True, 'status': 'cancelled', 'task_id': 'task',
                'execution_generation': kw['execution_generation'],
            })
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        monkeypatch.setattr(service, 'publish_terminal', lambda *args: None)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        action, = service.status('room')['pending_actions']
        assert action == {'kind': 'discard', 'member_id': 'one', 'task_id': 'task', 'execution_generation': 1}
        for change in ({'member_id': 'two'}, {'execution_generation': 2}, {'execution_generation': True}):
            with pytest.raises(RuntimeStoreError):
                service.discard_room_task(**{**args, **change})
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            service.retry_room_task(**args)
        assert not calls
        # A deferred attempt is explicit retryable work, but a canonical unknown
        # still blocks it; no generation advances on either refusal.
        tasks.defer_indeterminate_task(db.db_path, identity, recovered,
            expected_execution_generation=1, expected_cancel_generation=0,
            reason='test-deferred', clock=lambda: later)
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            service.retry_room_task(**args)
        rpc.info = lambda **kw: {'status': 'idle', 'active': False}
        retried = service.retry_room_task(**args)
        assert retried['status'] == 'queued'
        assert retried['execution_generation'] == 1
        tasks.start_task(db.db_path, identity, recovered, expected_cancel_generation=0, clock=lambda: later)
        later += 31
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)
        args['execution_generation'] = 2
        result = service.discard_room_task(**args)
        assert result['status'] == 'cancelled'
        assert calls[0]['execution_generation'] == 2
        assert calls[0]['expected_task_id'] == 'task'
        assert tasks.get_task(db.db_path, identity)['execution_generation'] == 2
        assert service.discard_room_task(**args)['status'] == 'cancelled'
        assert len(calls) == 1
        with pytest.raises(RuntimeStoreError):
            service.discard_room_task(**{**args, 'execution_generation': 1})


def test_unknown_discard_reservation_survives_lost_reply_without_holding_locks(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        runner = SimpleNamespace(_draining=False)
        authority = SimpleNamespace(
            db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'),
            instance_id='test', runner=runner,
        )
        runner.session_authority = authority
        service = CanonicalHostedRoomService(authority, None)
        authority.hosted_room_service = service
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
        ])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db.db_path, identity, payload={
            'target_profile': 'default', 'target_member_id': 'one',
            'source_event_seq': 1, 'prompt': 'frozen',
        }, clock=time.time)
        lease = tasks.acquire_lease(
            db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1,
            process_generation='test', ttl_seconds=30, clock=time.time,
        )
        tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        later = time.time() + 31
        monkeypatch.setattr(service.runtime, 'clock', lambda: later)
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)

        calls = []
        def discard(**params):
            acquired = service._policy_lock.acquire(blocking=False)
            assert acquired, 'target RPC inherited the source policy lock'
            service._policy_lock.release()
            with sqlite3.connect(db.db_path, timeout=0) as contender:
                contender.execute('BEGIN IMMEDIATE')
                contender.rollback()
            with db._read_ctx() as conn:
                rows = conn.execute(
                    "SELECT value FROM state_meta WHERE key LIKE 'gateway.hosted.discard_reservation.v1:%'"
                ).fetchall()
            assert len(rows) == 1 and json.loads(rows[0][0])['state'] == 'reserved'
            calls.append(dict(params))
            if len(calls) == 1:
                raise TimeoutError('inert lost target reply')
            return {
                'discarded': True, 'status': 'cancelled',
                'task_id': 'task', 'execution_generation': 1,
            }

        rpc = SimpleNamespace(ref=SimpleNamespace(session_id='session'), discard=discard)
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        monkeypatch.setattr(service, 'publish_terminal', lambda *args: None)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)

        with pytest.raises(TimeoutError, match='lost target reply'):
            service.discard_room_task(**args)
        assert tasks.get_task(db.db_path, identity)['status'] == 'indeterminate'
        result = service.discard_room_task(**args)
        assert result['status'] == 'cancelled' and len(calls) == 2
        with db._read_ctx() as conn:
            row = conn.execute(
                "SELECT value FROM state_meta WHERE key LIKE 'gateway.hosted.discard_reservation.v1:%'"
            ).fetchone()
        saved = json.loads(row[0])
        assert saved['state'] == 'completed'
        assert saved['target_result_digest']
