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


def test_a_peer_home_moves_on_only_from_an_unchanged_retired_attempt(hosted_owner, tmp_path):
    """Across owner sockets the source's 'retired' answer carries the committed input and owner,
    so a changed row or owner is still a hard denial and only an unchanged one moves the queue on."""
    import pytest
    from types import SimpleNamespace
    from gateway.control_socket import GatewayControlServer
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_contract import SessionRef
    from gateway.session_hosted_transport import (
        HostedRoomOwnerRPC, check_remote_hosted_admission, install_hosted_transport)
    authority, loop, _, _ = hosted_owner
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir(mode=0o700)
    target.mkdir(mode=0o700)
    authority.profile_id = str(target)
    answer = {'owner': 'room-owner', 'retired': False}

    def attest(selector, operation, params):
        result = {'owner': answer['owner'], 'target_home': authority.profile_id, 'prompt': 'input', 'attachments': []}
        return {**result, 'retired': True} if operation == 'execute' and answer['retired'] else result

    def server(home):
        return GatewayControlServer(home, verb_handlers={'identify': lambda: {
            'runtime_protocol': 1, 'state': 'ready', 'instance_id': 'test', 'authority_epoch': 1,
            'served_profiles': [{'home': str(home), 'profile_id': str(home)}],
            'capabilities': ['session-authority-v1'], 'api_origin': 'http://127.0.0.1:1', 'supervisor': 'none'}})
    servers = [server(source), server(target)]
    install_hosted_transport(servers[0], SimpleNamespace(profile_id=str(source)), loop, attest=attest)
    install_hosted_transport(servers[1], authority, loop, attest=lambda *a: None)
    for s in servers:
        assert asyncio.run_coroutine_threadsafe(s.start(), loop).result()
    try:
        rpc = HostedRoomOwnerRPC(home=target, source_home=source, room_id='room', member_id='member', profile='default')
        coords = dict(profile='default', source='bot_room')
        sid = rpc.create(**coords, title='Group: room')['session_id']
        rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room', 'task', 'thread', 'turn'),
                   execution_generation=1, member_id='member', on_terminal=lambda r: None)
        row, = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        ref = SessionRef(authority.profile_id, sid)
        answer['retired'] = True
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            check_remote_hosted_admission(authority, ref, row)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            check_remote_hosted_admission(authority, ref, {**row, 'payload': {'text': 'changed'}})
        answer['owner'] = 'someone-else'
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            check_remote_hosted_admission(authority, ref, row)
    finally:
        for s in servers:
            asyncio.run_coroutine_threadsafe(s.stop(), loop).result()


def _run_drain(authority, loop, ref):
    """Drive one drain to its first real claim; returns the payloads it reached, in order."""
    from gateway import session_authority
    claimed = []
    original = session_authority.SessionAuthority._claim_next

    async def claim(self, ref, live):
        result = await original(self, ref, live)
        claimed.append(result[1]['payload']['text'] if result and result[1] else None)
        raise asyncio.CancelledError  # stop at the first real claim: the head it reached is the proof

    async def drain():
        try:
            await authority._claim_with_retry(ref, authority.sessions[ref.session_id])
        except asyncio.CancelledError:
            pass

    session_authority.SessionAuthority._claim_next = claim
    try:
        asyncio.run_coroutine_threadsafe(drain(), loop).result(10)
    finally:
        session_authority.SessionAuthority._claim_next = original
    return claimed


def test_a_hosted_head_whose_member_home_is_gone_is_refused_not_paused(hosted_owner, monkeypatch):
    """#150885: a queued hosted turn whose member home is no longer served refuses every drain with
    ``permission_denied`` -- a reason the drain only ever paused behind, so the member's FIFO stayed
    stuck for good. That turn can never run: it is settled (its submitter still gets the refusal) and
    the queue is usable again once the member is served."""
    from gateway import hosted_room_driver as tasks
    authority, loop, _, _ = hosted_owner
    service, rpc, stale, _ = _queued_member_turn(hosted_owner, monkeypatch, 'stale', 'first ask')
    ref = rpc.ref  # both transport shapes carry one; the union does not declare it
    served = {'default': Path(authority.profile_id)}
    monkeypatch.setattr(service, 'profile_homes', dict)  # the member's home is no longer served

    _run_drain(authority, loop, ref)

    rows = {r['payload']['text']: (r['status'], r['outcome'])
            for r in list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)}
    assert rows['first ask'] == ('terminal', 'cancelled')
    # The FIFO is not paused behind it: the room stops that attempt and, once the member is served
    # again, its next turn is claimed.
    tasks.begin_task_cancel(authority.db.db_path, stale, cancel_id='stop-1', expected_cancel_generation=0,
                            clock=time.time)
    tasks.complete_task_cancel(authority.db.db_path, stale, cancel_id='stop-1', expected_cancel_generation=1,
                               clock=time.time)
    monkeypatch.setattr(service, 'profile_homes', lambda: served)
    _queued_member_turn(hosted_owner, monkeypatch, 'live', 'second ask')
    claimed = _run_drain(authority, loop, ref)
    assert claimed and claimed[-1] == 'second ask'


def test_a_queued_turn_whose_room_task_is_gone_reads_stale_not_denied(hosted_owner, monkeypatch):
    """#150885: the attempt's own row can be gone (a pruned terminal task, or a source that dropped it).
    That turn can never run, so the check says so -- ``stale_hosted_task`` -- instead of folding it into
    the hard denial the drain would pause behind. A changed input is still a hard denial."""
    import json
    import pytest
    from dataclasses import asdict
    from gateway import hosted_room_driver as tasks
    authority, _, _, _ = hosted_owner
    service, rpc, _, _ = _queued_member_turn(hosted_owner, monkeypatch, 'task', 'frozen')
    ref = rpc.ref
    row, = [r for r in list_session_admissions(authority.db, session_id=ref.session_id) if r['status'] == 'queued']
    gone = tasks.TaskIdentity('room', 'pruned', 'thread', 'pruned')
    stale = {**row, 'request_id': 'hosted:' + json.dumps([asdict(gone), 1], sort_keys=True, separators=(',', ':'))}
    with pytest.raises(RuntimeStoreError, match='stale_hosted_task'):
        service.check_admission(ref, stale)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        service.check_admission(ref, {**row, 'payload': {'text': 'changed'}})


def test_a_remote_execute_at_an_unserved_home_reads_stale_not_denied(hosted_owner, monkeypatch):
    """Across owner sockets the target asks ``attest``, not ``check_admission``: an unserved member home
    has to answer the same ``stale_hosted_task`` verdict, or the target's queued head pauses under
    ``permission_denied`` on every drain and the member's FIFO never moves. A new submit and a home that
    comes back somewhere else stay hard denials -- the second is a moved binding, not a dead attempt."""
    import pytest
    from dataclasses import asdict
    authority, _, _, _ = hosted_owner
    service, _, identity, _ = _queued_member_turn(hosted_owner, monkeypatch, 'unserved', 'pending ask')
    selector = {'room_id': 'room', 'member_id': 'one', 'profile': 'default'}
    params = {'_target_home': str(authority.profile_id), 'task': asdict(identity), 'execution_generation': 1}

    monkeypatch.setattr(service, 'profile_homes', dict)  # the member's bot is no longer served here
    with pytest.raises(RuntimeStoreError) as unserved:
        service.attest(selector, 'execute', params)
    assert unserved.value.reason == 'stale_hosted_task'

    with pytest.raises(RuntimeStoreError) as new_submit:
        service.attest(selector, 'submit', params)
    assert new_submit.value.reason == 'permission_denied'

    moved = Path(authority.profile_id).parent / 'moved-home'
    monkeypatch.setattr(service, 'profile_homes', lambda: {'default': moved})
    with pytest.raises(RuntimeStoreError) as rebinding:
        service.attest(selector, 'execute', params)
    assert rebinding.value.reason == 'permission_denied'
