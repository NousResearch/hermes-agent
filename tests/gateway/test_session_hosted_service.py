import asyncio
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, RuntimeStoreError


@pytest.mark.asyncio
async def test_room_owner_survives_service_restart_and_foreign_actor_cannot_claim(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
    try:
        from tui_gateway import hosted_room_service
        def forbidden_legacy(*args):
            raise AssertionError('canonical service constructed legacy RPC')
        monkeypatch.setattr(hosted_room_service, 'HostedRoomServerRPC', forbidden_legacy)
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        service.authorize_room('alice', 'room', create=True)
        from gateway.hosted_rooms import create_room
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id='test', members=[
            {'member_id': 'default', 'profile': 'default', 'handle': 'bot'},
            {'member_id': 'other', 'profile': 'other', 'handle': 'other'}])
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        assert service.authorize_room('alice', 'room') is True
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.authorize_room('bob', 'room', create=True)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.authorize_room('bob', 'room')
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.authorize_room('alice', 'absent')
        # A historical row without an owner cannot be adopted, including retired IDs.
        with db._lock:
            db._conn.execute("INSERT INTO hosted_room_retired_ids(room_id,retired_at) VALUES('retired',1)")
            db._conn.commit()
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.authorize_room('alice', 'retired', create=True)
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            service.authorize_room('alice', ' room ', create=True)
    finally:
        db.close()


def test_private_hosted_policy_restores_without_public_source_admission(tmp_path):
    from dataclasses import asdict, replace
    from gateway.session_policy import build_policy, restore_policy
    policy = replace(build_policy({'source': 'gui', 'cwd': str(tmp_path), 'model': 'fixture'}, {}),
                     source='bot_room', platform='bot_room')
    assert restore_policy(asdict(policy)).source == 'bot_room'
    with pytest.raises(RuntimeStoreError):
        build_policy({'source': 'bot_room'}, {})



def test_hosted_dequeue_checks_exact_task_member_and_frozen_input(tmp_path, monkeypatch):
    import json
    import time
    from dataclasses import asdict
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        payload = {'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}
        tasks.admit_task(db.db_path, identity, payload=payload, clock=time.time)
        lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1,
                                   process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        task, = tasks.list_tasks(db.db_path, room_id='room')
        rpc = service._resolve_member_transport(HostedRoomBinding('room', gateway, 1), task)
        from hermes_state_runtime import admit_session_input
        db.create_session(rpc.ref.session_id, source='cli')
        row = admit_session_input(db, epoch=authority.epoch, principal_id='alice', session_id=rpc.ref.session_id,
            request_id='hosted:' + json.dumps([asdict(identity), task['execution_generation']]), payload={'text': 'frozen'})
        assert service.check_admission(rpc.ref, row) == task
        for bad in ({**row, 'principal_id': 'bob'}, {**row, 'payload': {'text': 'changed'}},
                    {**row, 'request_id': 'hosted:' + json.dumps([asdict(identity), 999])}):
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                service.check_admission(rpc.ref, bad)
        with db._lock:
            db._conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room'", (json.dumps([
                {'member_id': 'replacement', 'profile': 'default', 'handle': 'replacement'},
                {'member_id': 'two', 'profile': 'other', 'handle': 'two'}]),))
            db._conn.commit()
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            service.check_admission(rpc.ref, row)



@pytest.mark.asyncio
async def test_service_does_not_plan_or_run_historical_unowned_rooms(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        for room_id in ('unowned', 'owned'):
            if room_id == 'owned':
                service.authorize_room('alice', room_id, create=True)
            create_room(db.db_path, room_id=room_id, name=room_id,
                authority_gateway_id=local_authority_gateway_id(), members=[
                    {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                    {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        assert [b.room_id for b in service.bindings()] == ['owned']
