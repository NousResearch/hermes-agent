"""A local Stop request is not evidence that its canonical producer ended."""
from pathlib import Path
import asyncio
import time

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state_runtime import (
    RuntimeStoreError, begin_runtime_epoch, claim_session_input, get_session_admission,
    recover_session_inputs, settle_session_input,
)
from tests.gateway.test_session_hosted_rpc import owner  # noqa: F401


@pytest.mark.parametrize('producer_state', ['queued', 'started', 'restart_unknown'])
def test_acknowledged_room_stop_waits_for_the_producer_terminal(owner, monkeypatch, producer_state):
    authority, loop, _, agent = owner
    service = CanonicalHostedRoomService(authority, loop)
    monkeypatch.setattr(service, 'profile_homes', lambda: {'default': Path(authority.profile_id), 'other': Path(authority.profile_id)})
    service.authorize_room('alice', 'room', create=True)
    gateway = rooms.local_authority_gateway_id()
    rooms.create_room(authority.db.db_path, room_id='room', name='Room', authority_gateway_id=gateway,
        members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                 {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
    service.send(room_id='room', event_id='input', payload={'text': '@one input', 'thread_id': 'thread'})
    task, = tasks.list_tasks(authority.db.db_path, room_id='room', status='queued')
    identity = task['identity']
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    tasks.start_task(authority.db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
    member = service._resolve_member_transport(binding, tasks.get_task(authority.db.db_path, identity))
    coords = dict(profile='default', source='bot_room')
    sid = member.create(**coords, title='Group: room')['session_id']
    receipt = member.submit(**coords, session_id=sid, prompt=task['payload']['prompt'], task=identity,
        execution_generation=1, on_terminal=lambda _: None)
    if producer_state != 'queued':
        claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    if producer_state == 'restart_unknown':
        authority.epoch = begin_runtime_epoch(authority.db, instance_id='restarted-owner')
        recover_session_inputs(authority.db, epoch=authority.epoch)
    refusal = None
    try:
        service.stop_room('room', cancel_id='stop', require_acknowledged=True)
    except RuntimeError as exc:
        refusal = str(exc)
    canonical = get_session_admission(authority.db, admission_id=receipt['admission_id'])
    hosted = tasks.get_task(authority.db.db_path, identity)
    if producer_state == 'queued':
        assert refusal is None and hosted['status'] == 'cancelled'
        assert (canonical['status'], canonical['outcome']) == ('terminal', 'cancelled')
        assert not agent.interrupted
        return
    if producer_state == 'restart_unknown':
        assert canonical['status'] == 'unknown'
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            member.interrupt(**coords, session_id=sid, expected_task_id=identity.task_id, expected_execution_generation=1)
        assert hosted['status'] == 'stopping' and refusal is not None, {
            'hosted_status': hosted['status'], 'canonical_status': canonical['status'], 'refusal': refusal}
        # A separately acknowledged unknown outcome is durable terminal evidence.
        asyncio.run_coroutine_threadsafe(authority.resolve_unknown(
            member.principal, member.ref, canonical['admission_id'], canonical['generation']), loop).result(10)
        service.stop_room('room', cancel_id='stop', require_acknowledged=True)
        assert tasks.get_task(authority.db.db_path, identity)['status'] == 'cancelled'
        return
    assert agent.interrupted and canonical['status'] == 'started'
    assert hosted['status'] == 'stopping' and refusal is not None, {
        'hosted_status': hosted['status'], 'canonical_status': canonical['status'], 'refusal': refusal}
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=canonical['admission_id'],
        generation=canonical['generation'], outcome='interrupted',
        result={'result': {'interrupted': True, 'final_response': ''}, 'usage': {}})
    service.stop_room('room', cancel_id='stop', require_acknowledged=True)
    assert tasks.get_task(authority.db.db_path, identity)['status'] == 'cancelled'
