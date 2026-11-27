import asyncio
from types import SimpleNamespace

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.gateway.hosted_rooms import HostedOwnerClaim, HostedTaskIdentity
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_room_owner_survives_service_restart_and_foreign_actor_cannot_claim(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = open_persistence(profile_scope(home=tmp_path))
    authority = SimpleNamespace(db=db, persistence=storage_handle(tmp_path), profile_id=str(tmp_path), epoch=db.runtime.begin_epoch(instance_id='test'))
    try:
        from tui_gateway import hosted_room_service
        def forbidden_legacy(*args):
            raise AssertionError('canonical service constructed legacy RPC')
        monkeypatch.setattr(hosted_room_service, 'HostedRoomServerRPC', forbidden_legacy)
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        members = [{'member_id': 'default', 'profile': 'default', 'handle': 'bot'},
                   {'member_id': 'other', 'profile': 'other', 'handle': 'other'}]
        authority.persistence.hosted_rooms.create_room(
            placement='profile', room_id='room', name='Room', authority_gateway_id='test', members=members,
            owner=HostedOwnerClaim(epoch=authority.epoch, actor='alice', rule=None))
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        assert service.authorize_room('alice', 'room') is True
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            service.insert_room(room_id='room', name='Room', members=members, owner='bob')
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            service.authorize_room('bob', 'room')
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            service.authorize_room('alice', 'absent')
        # A historical row without an owner cannot be adopted, including retired IDs.
        raw_holder(db)._execute_write(lambda conn: conn.execute(
            "INSERT INTO hosted_room_retired_ids(room_id,retired_at) VALUES('retired',1)"))
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            service.insert_room(room_id='retired', name='Room', members=members, owner='alice')
        with pytest.raises(AuthorityRefusal, match='invalid_params'):
            service.insert_room(room_id=' room ', name='Room', members=members, owner='alice')
    finally:
        db.close()


def test_private_hosted_policy_restores_without_public_source_admission(tmp_path):
    from dataclasses import asdict, replace
    from gateway.session_policy import build_policy, restore_policy
    policy = replace(build_policy({'source': 'gui', 'cwd': str(tmp_path), 'model': 'fixture'}, {}),
                     source='bot_room', platform='bot_room')
    assert restore_policy(asdict(policy)).source == 'bot_room'
    with pytest.raises(AuthorityRefusal):
        build_policy({'source': 'bot_room'}, {})



def test_hosted_dequeue_checks_exact_task_member_and_frozen_input(tmp_path, monkeypatch):
    import json
    import time
    from dataclasses import asdict
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import local_authority_gateway_id
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(db=db, persistence=storage_handle(tmp_path), profile_id=str(tmp_path), epoch=db.runtime.begin_epoch(instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        gateway = local_authority_gateway_id()
        service.insert_room(room_id='room', name='Room', owner='alice', members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
        payload = {'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}
        tasks.admit_task(db, identity, placement='profile', payload=payload, clock=time.time)
        lease = tasks.acquire_lease(db, placement='profile', room_id='room', gateway_id=gateway, authority_epoch=1,
                                   process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db, identity, lease, placement='profile', expected_cancel_generation=0, clock=time.time)
        task, = tasks.list_tasks(db, placement='profile', room_id='room')
        rpc = service._resolve_member_transport(HostedRoomBinding('room', gateway, 1), task)
        row = {'principal_id': 'alice', 'request_id': 'hosted:' + json.dumps([asdict(identity), task['execution_generation']]),
               'payload': {'text': 'frozen'}}
        assert service.check_admission(rpc.ref, row) == task
        for bad in ({**row, 'principal_id': 'bob'}, {**row, 'payload': {'text': 'changed'}},
                    {**row, 'request_id': 'hosted:' + json.dumps([asdict(identity), 999])}):
            with pytest.raises(AuthorityRefusal, match='permission_denied'):
                service.check_admission(rpc.ref, bad)
        raw_holder(db)._execute_write(lambda conn: conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room'", (json.dumps([
            {'member_id': 'replacement', 'profile': 'default', 'handle': 'replacement'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}]),)))
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            service.check_admission(rpc.ref, row)



@pytest.mark.asyncio
async def test_service_does_not_plan_or_run_historical_unowned_rooms(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(db=db, persistence=storage_handle(tmp_path), profile_id=str(tmp_path), epoch=db.runtime.begin_epoch(instance_id='test'))
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                   {'member_id': 'two', 'profile': 'other', 'handle': 'two'}]
        create_room(db, placement='profile', room_id='unowned', name='unowned',
                    authority_gateway_id=local_authority_gateway_id(), members=members)
        service.insert_room(room_id='owned', name='owned', members=members, owner='alice')
        assert [b.room_id for b in service.bindings()] == ['owned']



@pytest.mark.parametrize('driver', [False, True])
def test_rejected_create_never_leaves_owner_only_reservation(tmp_path, monkeypatch, driver):
    """Owner and room commit together: no failed create may fence a later valid creator."""
    import sqlite3
    from gateway.session_controls import AuthorityConnection
    from tests.fixtures.storage_holder import hosted_room_reload
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'profiles' / 'two').mkdir(parents=True)
    (tmp_path / 'config.yaml').write_text(
        'hosted_rooms:\n  profiles:\n    two: ' + str((tmp_path / 'profiles' / 'two').resolve()) + '\n')
    members = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
               {'member_id': 'two', 'profile': 'two', 'handle': 'two'}]
    with open_persistence(profile_scope(home=tmp_path)) as db:
        authority = SimpleNamespace(db=db, persistence=storage_handle(tmp_path), profile_id=str(tmp_path), instance_id='owner', events={},
                                    epoch=db.runtime.begin_epoch(instance_id='test'))
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, None)
        if driver:
            monkeypatch.setattr(service.runtime, 'status', lambda: {'running': True, 'stopping': False})
        alice, bob, carol = (AuthorityConnection(authority, object(), {'user_id': u})
                             for u in ('alice', 'bob', 'carol'))

        async def create(connection, room_id, **params):
            reply = await connection.dispatch({'id': 1, 'method': 'groups.create',
                                               'params': {'room_id': room_id, 'name': 'Room',
                                                          'members': members, **params}})
            return reply.get('error', {}).get('message') or 'ok'

        def counts():
            with raw_holder(db)._read_ctx() as conn:
                return (conn.execute("SELECT COUNT(*) FROM state_meta WHERE key LIKE 'gateway.hosted.owner.v1:%'").fetchone()[0],
                        conn.execute('SELECT COUNT(*) FROM hosted_rooms').fetchone()[0])

        async def probe():
            invalid = [{'members': []}, {'name': ''},
                       {'members': [members[0], {**members[1], 'target': {'kind': 'local', 'profile': 'one'}}]}]
            for index, params in enumerate(invalid):
                assert await create(alice, f'bad-{index}', **params) == 'invalid_params'
            real_reload = hosted_room_reload()
            def storage_failure(*args):
                raise sqlite3.OperationalError('disk I/O error')
            monkeypatch.setattr('persistence.sqlite.gateway.hosted_rooms._reload', storage_failure)
            assert await create(alice, 'bad-3') == 'storage_unavailable'
            monkeypatch.setattr('persistence.sqlite.gateway.hosted_rooms._reload', real_reload)
            assert counts() == (0, 0)
            # Every rejected ID stays claimable by any actor; an exact retry stays idempotent.
            for index in range(4):
                assert await create(bob, f'bad-{index}') == 'ok'
                assert await create(bob, f'bad-{index}') == 'ok'
                assert await create(alice, f'bad-{index}') == 'permission_denied'
            # First-writer fencing under a real race: exactly one valid creator wins.
            raced = await asyncio.gather(*(create(c, 'raced') for c in (alice, carol)))
            assert sorted(raced) == ['ok', 'permission_denied'], raced
            assert counts() == (5, 5)
        asyncio.run(probe())
