"""A per-member hosted producer refuses a submit that names another member."""
import time
from pathlib import Path

import pytest
from persistence.gateway.hosted_rooms import HostedTaskIdentity


def test_member_producer_admits_its_own_member_and_refuses_another(hosted_owner):
    from persistence.gateway.hosted_rooms import HostedTaskIdentity
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from persistence.core.errors import AuthorityRefusal
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default',
                                 principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    args = dict(**coords, session_id=sid, prompt='input', task=HostedTaskIdentity('room', 'task', 'thread', 'turn'),
                execution_generation=1, on_terminal=lambda receipt: None)
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        rpc.submit(**args, member_id='other')
    assert authority.persistence.admissions.list(session_id=sid, pending_only=False) == []
    receipt = rpc.submit(**args, member_id='member')
    rows = authority.persistence.admissions.list(session_id=sid, pending_only=False)
    assert [row.admission_id for row in rows] == [receipt['admission_id']]


def test_canonical_resolver_binds_the_member_the_driver_names(hosted_owner, monkeypatch):
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import local_authority_gateway_id
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from persistence.core.errors import AuthorityRefusal
    from tui_gateway.hosted_room_driver import HostedRoomBinding, _member_id
    authority, loop, _, _ = hosted_owner
    service = CanonicalHostedRoomService(authority, loop)
    monkeypatch.setattr(service, 'profile_homes', lambda: {'default': Path(authority.profile_id)})
    gateway = local_authority_gateway_id()
    service.insert_room(room_id='room', name='Room', owner='alice', members=[
        {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
        {'member_id': 'two', 'profile': 'default', 'handle': 'two'}])
    identity = HostedTaskIdentity('room', 'task', 'thread', 'turn')
    tasks.admit_task(authority.persistence, identity, placement='profile', clock=time.time, payload={
        'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'})
    lease = tasks.acquire_lease(authority.persistence, placement='profile', room_id='room', gateway_id=gateway,
        authority_epoch=1, process_generation='test', ttl_seconds=120, clock=time.time)
    tasks.start_task(authority.persistence, identity, lease, placement='profile', expected_cancel_generation=0, clock=time.time)
    task, = tasks.list_tasks(authority.persistence, placement='profile', room_id='room')
    rpc = service._resolve_member_transport(HostedRoomBinding('room', gateway, 1), task)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    args = dict(**coords, session_id=sid, prompt='frozen', task=identity,
                execution_generation=task['execution_generation'], on_terminal=lambda receipt: None)
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        rpc.submit(**args, member_id='two')
    assert authority.persistence.admissions.list(session_id=sid, pending_only=False) == []
    receipt = rpc.submit(**args, member_id=_member_id(task))
    rows = authority.persistence.admissions.list(session_id=sid, pending_only=False)
    assert [row.admission_id for row in rows] == [receipt['admission_id']]
