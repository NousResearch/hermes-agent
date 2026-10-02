"""An Allow never releases current work after its room authority is withdrawn."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import sqlite3
import threading
import time

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state_runtime import RuntimeStoreError, claim_session_input, settle_session_input
from tests.gateway.test_session_hosted_rpc import owner  # noqa: F401
from tests.gateway.test_session_group_peer_controls import case, rpc, selector  # noqa: F401
from tui_gateway.hosted_room_driver import HostedRoomBinding


def _quarantine_room(authority):
    with authority.db._read_ctx() as conn:
        installed = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                 "AND name='hosted_room_quarantine'").fetchone()
    if installed is None:
        pytest.skip('Requires the composed #99107 hosted_room_quarantine schema')
    authority.db._execute_write(lambda conn: conn.execute(
        "INSERT INTO hosted_room_quarantine VALUES('room','unsafe_authority_demotion',?)", (time.time(),)))
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(authority.db.db_path, room_id='room')


def _approval_waiting_for_writer(authority, loop, member_rpc, sid, monkeypatch):
    requested, locked, release, progressed = (threading.Event() for _ in range(4))
    context = ContextVar('approval-test-owner', default=None)
    original = member_rpc.authorizer

    def authorize(operation, identity, generation):
        if operation == 'approve':
            assert context.get() == 'owned'
            requested.set()
            assert locked.wait(10), 'writer did not acquire the transaction'
        return original(operation, identity, generation)

    def writer():
        assert requested.wait(10), 'approval never reached its final authorization'
        with sqlite3.connect(authority.db.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute("UPDATE hosted_room_driver_tasks SET status='stopping' WHERE room_id='room'")
            locked.set()
            assert release.wait(10), 'test did not release the real writer'

    def approve():
        token = context.set('owned')
        try:
            return member_rpc.approve(session_id=sid, request_id='approval-after-barrier', choice='once',
                expected_task_id='task', expected_execution_generation=1)
        finally:
            context.reset(token)

    monkeypatch.setattr(member_rpc, 'authorizer', authorize)
    with ThreadPoolExecutor(max_workers=2) as workers:
        writing = workers.submit(writer)
        pending = workers.submit(approve)
        try:
            assert locked.wait(10)
            loop.call_soon_threadsafe(progressed.set)
            assert progressed.wait(2), 'approval writer wait blocked the owning event loop'
        finally:
            release.set()
        writing.result(timeout=10)
        return pending.result(timeout=10)


def _approval_after_new_task(authority, service, member_rpc, sid, row, attempt, effect, choice, monkeypatch):
    handed_off, release = threading.Event(), threading.Event()
    original = member_rpc._call

    def delayed(operation, **params):
        if operation == 'approve':
            handed_off.set()
            assert release.wait(10), 'test did not deliver the old approval'
        return original(operation, **params)

    monkeypatch.setattr(member_rpc, '_call', delayed)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(service.approve_room_task, 'room', member_id='one', task_id='task',
            execution_generation=1, request_id='approval-after-barrier', choice=choice)
        try:
            assert handed_off.wait(10), 'source did not authorize the old selection'
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                generation=row['generation'], outcome='completed',
                result={'result': {'final_response': 'old complete'}, 'usage': {}})
            tasks.settle_task(authority.db.db_path, attempt, settlement_id='old-complete', status='settled',
                result={'text': 'old complete'}, clock=time.time)
            event = rooms.append_event(authority.db.db_path, room_id='room', event_id='second-input', kind='message.user',
                actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'second', 'thread_id': 'thread'},
                authority_gateway_id=attempt.lease.gateway_id, authority_epoch=attempt.lease.authority_epoch)
            identity = tasks.TaskIdentity('room', 'later-task', 'thread', 'later-turn')
            tasks.admit_task(authority.db.db_path, identity, payload={'target_profile': 'default',
                'target_member_id': 'one', 'source_event_seq': event['seq'], 'prompt': 'second'}, clock=time.time)
            tasks.start_task(authority.db.db_path, identity, attempt.lease, expected_cancel_generation=0, clock=time.time)
            member_rpc.submit(profile='default', source='bot_room', session_id=sid, prompt='second', task=identity,
                execution_generation=1, on_terminal=lambda _: None)
            second = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            assert second['admission_id'] != row['admission_id'] and second['generation'] > row['generation']
            live = authority.sessions[sid]
            authority.register_approval(sid, second['generation'], live.route,
                {'request_id': 'approval-after-barrier', 'command': 'later operation'})
            live.controls.remote_responders['approval-after-barrier'] = lambda *answer: effect.write_text(repr(answer))
            service._pending_actions[('room', 'one')] = dict(kind='approval', task_id='later-task',
                execution_generation=1, request_id='approval-after-barrier',
                approval={'choices': ['once', 'deny']}, session_id=sid)
        finally:
            release.set()
        return pending.result(timeout=10)


@pytest.mark.parametrize('barrier,choice,entry', [
    (barrier, choice, entry)
    for barrier in ['healthy', 'stop_event', 'stop', 'epoch', 'retiring']
    for choice in ['once', 'deny'] for entry in ['service', 'producer']
] + [('writer_stop', 'once', 'producer'), ('quarantine', 'once', 'service'),
     ('quarantine', 'once', 'producer'), ('later_task', 'once', 'service'), ('later_task', 'deny', 'service')])
def test_local_pending_approval_after_durable_barrier(owner, monkeypatch, tmp_path, barrier, choice, entry):
    authority, loop, _, _ = owner
    service = CanonicalHostedRoomService(authority, loop)
    monkeypatch.setattr(service, 'profile_homes', lambda: {'default': Path(authority.profile_id)})
    service.authorize_room('alice', 'room', create=True)
    gateway = rooms.local_authority_gateway_id()
    rooms.create_room(authority.db.db_path, room_id='room', name='Room', authority_gateway_id=gateway,
        members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'}])
    event = rooms.append_event(authority.db.db_path, room_id='room', event_id='input', kind='message.user',
        actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'frozen', 'thread_id': 'thread'},
        authority_gateway_id=gateway, authority_epoch=1)
    identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
    tasks.admit_task(authority.db.db_path, identity, payload={
        'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': event['seq'], 'prompt': 'frozen'},
        clock=time.time)
    lease = tasks.acquire_lease(authority.db.db_path, room_id='room', gateway_id=gateway,
        authority_epoch=1, process_generation='audit', ttl_seconds=120, clock=time.time)
    attempt = tasks.start_task(authority.db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
    task = tasks.get_task(authority.db.db_path, identity)
    member_rpc = service._resolve_member_transport(HostedRoomBinding('room', gateway, 1), task)
    coords = {'profile': 'default', 'source': 'bot_room'}
    sid = member_rpc.create(**coords, title='Group: room')['session_id']
    member_rpc.submit(**coords, session_id=sid, prompt='frozen', task=identity,
        execution_generation=1, on_terminal=lambda _: None)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    live = authority.sessions[sid]
    authority.register_approval(sid, row['generation'], live.route,
        {'request_id': 'approval-after-barrier', 'command': 'audit control effect'})
    effect = tmp_path / 'approval-effect'
    live.controls.remote_responders['approval-after-barrier'] = lambda *answer: effect.write_text(repr(answer))
    action = member_rpc.info(**coords, session_id=sid)['pending_approval']
    service._pending_actions[('room', 'one')] = dict(kind='approval', task_id='task',
        execution_generation=1, request_id=action['request_id'], approval=action, session_id=sid)
    if barrier in {'stop', 'stop_event'}:
        rooms.request_room_stop(authority.db.db_path, room_id='room', cancel_id='audit-stop',
            expected_gateway_id=gateway, expected_epoch=1)
    if barrier == 'stop':
        tasks.begin_task_cancel(authority.db.db_path, identity, cancel_id='audit-stop',
            expected_cancel_generation=0, clock=time.time)
        assert tasks.get_task(authority.db.db_path, identity)['status'] == 'stopping'
    elif barrier == 'epoch':
        authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='room'"))
    elif barrier == 'retiring':
        service.begin_disband('room')
    elif barrier == 'quarantine':
        _quarantine_room(authority)
    result = None
    try:
        if barrier == 'writer_stop':
            result = _approval_waiting_for_writer(authority, loop, member_rpc, sid, monkeypatch)
        elif barrier == 'later_task':
            result = _approval_after_new_task(authority, service, member_rpc, sid, row, attempt, effect, choice, monkeypatch)
        elif entry == 'producer':
            result = member_rpc.approve(session_id=sid, request_id='approval-after-barrier', choice=choice,
                expected_task_id='task', expected_execution_generation=1)
        else:
            result = service.approve_room_task('room', member_id='one', task_id='task',
                execution_generation=1, request_id='approval-after-barrier', choice=choice)
    except (RuntimeStoreError, rooms.HostedRoomError, tasks.DriverStateError):
        pass
    if barrier == 'healthy' or (choice == 'deny' and barrier in {'stop_event', 'stop', 'retiring'}):
        assert result['status'] == 'resolved' and effect.is_file()
    else:
        assert not effect.exists(), (barrier, result, effect.read_text())
        if barrier == 'quarantine':
            assert result is None and ('room', 'one') in service._pending_actions


@pytest.mark.parametrize('barrier,choice', [
    (barrier, choice) for barrier in ['healthy', 'stop_event', 'stop', 'epoch', 'retiring']
    for choice in ['once', 'deny']
] + [('quarantine', 'once')])
def test_canonical_peer_rpc_after_durable_barrier(case, tmp_path, barrier, choice):
    c = case
    binding = c.service.bindings()[0]
    lease = c.service.runtime._ensure_lease(binding)
    tasks.start_task(c.service.db_path, c.original['identity'], lease,
        expected_cancel_generation=0, clock=c.service.runtime.clock)
    exact = selector(c)
    effect = tmp_path / 'peer-approval-effect'
    def accept_approval(**kwargs):
        effect.write_text(repr(kwargs))
        return {'resolved': 1}
    c.peer.approve_receipt = accept_approval
    c.service._pending_actions[('room', 'peer')] = dict(kind='approval',
        task_id=exact['task_id'], execution_generation=exact['execution_generation'],
        request_id='real-pending-control', approval={'choices': ['once', 'deny']})
    if barrier in {'stop', 'stop_event'}:
        rooms.request_room_stop(c.service.db_path, room_id='room', cancel_id='audit-stop',
            expected_gateway_id=binding.gateway_id, expected_epoch=1)
    if barrier == 'stop':
        tasks.begin_task_cancel(c.service.db_path, c.original['identity'], cancel_id='audit-stop',
            expected_cancel_generation=0, clock=c.service.runtime.clock)
        assert tasks.get_task(c.service.db_path, c.original['identity'])['status'] == 'stopping'
    elif barrier == 'epoch':
        c.authority.db._execute_write(lambda conn: conn.execute(
            "UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='room'"))
    elif barrier == 'retiring':
        c.service.begin_disband('room')
    elif barrier == 'quarantine':
        _quarantine_room(c.authority)
    reply = rpc(c, 'groups.approve', dict(**exact, request_id='real-pending-control', choice=choice))
    if barrier == 'healthy' or choice == 'deny':
        assert 'result' in reply and effect.is_file()
    else:
        assert not effect.exists(), (barrier, reply)
        if barrier == 'quarantine':
            assert 'error' in reply and ('room', 'peer') in c.service._pending_actions
