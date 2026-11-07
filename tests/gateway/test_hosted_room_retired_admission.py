"""A group member's queue never sticks behind the input of a turn the room has moved past.

Regression for #146567: the room deferred a member's turn while its input still sat queued at the
head of the member's session. Every later drain refused that head with ``permission_denied`` and
paused, so each new message to that member was saved and never answered.
"""
import asyncio
import time
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError, list_session_admissions


def _queued_member_turn(hosted_owner, monkeypatch, task_id, prompt):
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from tui_gateway.hosted_room_driver import HostedRoomBinding, _member_id
    authority, loop, _, _ = hosted_owner
    service = getattr(authority, 'hosted_room_service', None)
    if service is None:
        service = authority.hosted_room_service = CanonicalHostedRoomService(authority, loop)
        monkeypatch.setattr(service, 'profile_homes', lambda: {'default': Path(authority.profile_id)})
        service.authorize_room('alice', 'room', create=True)
        create_room(authority.db.db_path, room_id='room', name='Room', authority_gateway_id=local_authority_gateway_id(),
                    members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'}])
    gateway = local_authority_gateway_id()
    identity = tasks.TaskIdentity('room', task_id, 'thread', task_id)
    tasks.admit_task(authority.db.db_path, identity, clock=time.time, payload={
        'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': prompt})
    lease = tasks.acquire_lease(authority.db.db_path, room_id='room', gateway_id=gateway,
        authority_epoch=1, process_generation='test', ttl_seconds=120, clock=time.time)
    tasks.start_task(authority.db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
    task = tasks.get_task(authority.db.db_path, identity)
    rpc = service._resolve_member_transport(HostedRoomBinding('room', gateway, 1), task)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']  # the second call resumes the same session
    rpc.submit(**coords, session_id=sid, prompt=prompt, task=identity, member_id=_member_id(task),
               execution_generation=task['execution_generation'], on_terminal=lambda receipt: None)
    return service, rpc, identity, lease


def test_a_deferred_turn_s_queued_input_is_refused_and_the_next_turn_runs(hosted_owner, monkeypatch):
    from gateway import hosted_room_driver as tasks, session_authority
    authority, loop, _, _ = hosted_owner
    _, rpc, stale, lease = _queued_member_turn(hosted_owner, monkeypatch, 'stale', 'first ask')
    # The member is busy past the room's deadline: the room fences that turn as deferred.
    later = time.time() + 121
    recovered = tasks.acquire_lease(authority.db.db_path, room_id='room', gateway_id=lease.gateway_id,
        authority_epoch=1, process_generation='after', ttl_seconds=120, clock=lambda: later)
    tasks.recover_room(authority.db.db_path, recovered, clock=lambda: later)
    tasks.defer_indeterminate_task(authority.db.db_path, stale, recovered, expected_execution_generation=1,
        expected_cancel_generation=0, reason='member_unavailable', clock=lambda: later)
    tasks.release_lease(authority.db.db_path, recovered, clock=lambda: later)
    _queued_member_turn(hosted_owner, monkeypatch, 'live', 'second ask')

    claimed = []
    original = session_authority.SessionAuthority._claim_next

    async def claim(self, ref, live):
        result = await original(self, ref, live)
        claimed.append(result[1]['payload']['text'] if result and result[1] else None)
        raise asyncio.CancelledError  # stop at the first real claim: the head it reached is the proof
    monkeypatch.setattr(session_authority.SessionAuthority, '_claim_next', claim)

    async def drain():
        live = authority.sessions[rpc.ref.session_id]
        try:
            await authority._claim_with_retry(rpc.ref, live)
        except asyncio.CancelledError:
            pass
    asyncio.run_coroutine_threadsafe(drain(), loop).result(10)

    rows = {r['payload']['text']: (r['status'], r['outcome'])
            for r in list_session_admissions(authority.db, session_id=rpc.ref.session_id, pending_only=False)}
    assert rows['first ask'] == ('terminal', 'cancelled')
    assert claimed and claimed[-1] == 'second ask'


def test_a_running_turn_s_changed_input_is_still_a_hard_denial(hosted_owner, monkeypatch):
    """Only a retired attempt is refused as moved-on: a live one with altered input stays denied."""
    import pytest
    service, rpc, identity, _ = _queued_member_turn(hosted_owner, monkeypatch, 'task', 'frozen')
    authority = hosted_owner[0]
    row, = [r for r in list_session_admissions(authority.db, session_id=rpc.ref.session_id) if r['status'] == 'queued']
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        service.check_admission(rpc.ref, {**row, 'payload': {'text': 'changed'}})
    assert service.check_admission(rpc.ref, row)['identity'] == identity
