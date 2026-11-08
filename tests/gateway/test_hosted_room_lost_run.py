"""A turn lost in a restart must not leave its bot's queue paused, with the real authority behind the
driver: on this gateway, through another profile's daemon, and through the gateway's own controls."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway import hosted_room_context as context
from gateway import hosted_room_driver as state, hosted_rooms
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from gateway.session_hosted_service import CanonicalHostedRoomService
from gateway.session_hosted_transport import HostedRoomOwnerRPC
from gateway.session_settlement_recovery import _mark_unknown
from hermes_state_runtime import (
    begin_runtime_epoch, claim_session_input, list_session_admissions, recover_session_inputs)
from tests.gateway.test_hosted_observation_bounded import _owner_pair
from tui_gateway.hosted_room_driver import HostedRoomBinding, HostedRoomRuntime

ROOM = 'room'
MEMBERS = [{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
           {'member_id': 'two', 'profile': 'other', 'handle': 'two'}]
COORDS = {'profile': 'default', 'source': 'bot_room'}


def _restart(authority, session_id):
    """The bot was running when its gateway restarted: its admission recovers ``unknown``."""
    claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='after-restart')
    recover_session_inputs(authority.db, epoch=authority.epoch)


def _admissions(authority, session_id):
    return [row['status'] for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)]


def _pass_until(runtime, now, db, identity, statuses, limit=30):
    for _ in range(limit):
        runtime._stop.clear()
        runtime._run_room_once(runtime._rooms_provider()[0])
        if state.get_task(db, identity)['status'] in statuses:
            return
        now[0] += 3
    raise AssertionError(state.get_task(db, identity)['status'], runtime._last_error)


def _driver_runtime(db, rpc, now):
    runtime = HostedRoomRuntime(
        db_path=db, rooms=[HostedRoomBinding(ROOM, 'gateway-a', 1)], rpc=None, transport_resolver=lambda *_: rpc,
        turn_lock=lambda _: nullcontext(), clock=lambda: now[0], indeterminate_defer_seconds=5,
        active_poll_interval_seconds=0.01)
    # What CanonicalHostedRoomService wires for a member this gateway serves.
    runtime.confirm_deferred_stop = runtime.deferred_stop_confirmed
    runtime._wake.wait = lambda _timeout: runtime._stop.set()
    return runtime


@pytest.fixture(params=['this_gateway', 'another_daemon'])
def lost(request, hosted_owner, tmp_path):
    """A turn admitted before a restart, with a new process's driver about to look after it."""
    authority, loop, principal, _ = hosted_owner
    servers = []

    def build(*, routing=False, restart=_restart):
        db, now = authority.db.db_path, [100.0]
        hosted_rooms.create_room(db, room_id=ROOM, name='Batch', authority_gateway_id='gateway-a', members=MEMBERS)
        first = state.TaskIdentity(ROOM, 'first', 'thread', 'turn-1')
        state.admit_task(db, first, clock=lambda: now[0], payload={
            'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'input',
            **({'lead_routing': True} if routing else {})})
        old = state.acquire_lease(db, room_id=ROOM, gateway_id='gateway-a', authority_epoch=1,
                                  process_generation='before-restart', ttl_seconds=30, clock=lambda: now[0])
        state.start_task(db, first, old, expected_cancel_generation=0, clock=lambda: now[0])
        if request.param == 'this_gateway':
            rpc = HostedRoomAuthorityRPC(authority, loop, room_id=ROOM, member_id='one', profile='default',
                                         principal=principal, authorize=lambda *args: True)
        else:
            source, target, started = _owner_pair(hosted_owner, tmp_path)
            servers.extend(started)
            rpc = HostedRoomOwnerRPC(home=target, source_home=source, room_id=ROOM, member_id='one', profile='default')
        sid = rpc.create(**COORDS, title=f'Group: {ROOM}')['session_id']
        rpc.submit(**COORDS, session_id=sid, prompt='input', task=first, execution_generation=1,
                   on_terminal=lambda receipt: None, member_id='one')
        restart(authority, sid)
        rpc.callbacks.clear()  # they died with the process that registered them
        assert rpc.info(**COORDS, session_id=sid)['status'] == 'unknown'
        now[0] += 31  # the dead process's lease has lapsed
        return SimpleNamespace(db=db, now=now, first=first, rpc=rpc, sid=sid, authority=authority,
                               runtime=_driver_runtime(db, rpc, now))

    yield build
    for server in servers:
        asyncio.run_coroutine_threadsafe(server.stop(), loop).result()


def _next_turn(room, task_id='next'):
    identity = state.TaskIdentity(ROOM, task_id, 'thread', f'turn-{task_id}')
    state.admit_task(room.db, identity, clock=lambda: room.now[0], payload={
        'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 2, 'prompt': 'input'})
    return identity


@pytest.mark.parametrize('routing', [False, True], ids=['member', 'lead'])
def test_a_lost_turn_frees_its_bots_queue(lost, routing):
    room = lost(routing=routing)
    later = _next_turn(room)
    _pass_until(room.runtime, room.now, room.db, room.first, {'cancelled', 'deferred'})
    assert state.get_task(room.db, room.first)['status'] == ('cancelled' if routing else 'deferred')
    _pass_until(room.runtime, room.now, room.db, later, {'running'})
    assert _admissions(room.authority, room.sid) == ['terminal', 'queued']
    assert room.rpc.info(**COORDS, session_id=room.sid)['active'] is True


def test_an_answer_the_lost_run_left_behind_settles_its_turn(lost):
    def settlement_write_failed(authority, session_id):
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': 'The completed answer must survive.', 'completed': True}, 'usage': {}}
        assert _mark_unknown(authority, row) == 'unknown'

    room = lost(restart=settlement_write_failed)
    _pass_until(room.runtime, room.now, room.db, room.first, {'settled', 'deferred'})
    task = state.get_task(room.db, room.first)
    assert (task['status'], task['result'].get('text')) == ('settled', 'The completed answer must survive.')


def test_a_deferred_reply_survives_the_loss_of_its_daemon_monitor(lost):
    def captured_answer(authority, session_id):
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': 'Recovered from storage.', 'completed': True}, 'usage': {}}
        assert _mark_unknown(authority, row) == 'unknown'

    room = lost(restart=captured_answer)
    lease = room.runtime._ensure_lease(room.runtime._rooms_provider()[0])
    state.recover_room(room.db, lease, clock=room.runtime.clock)
    state.defer_indeterminate_task(room.db, room.first, lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason='member_unavailable', clock=room.runtime.clock)
    # Resolution persists the answer without a callback to the restarted room driver.
    room.rpc.discard(**COORDS, session_id=room.sid, expected_task_id=room.first.task_id, execution_generation=1)
    room.runtime._run_room_once(room.runtime._rooms_provider()[0])
    task = state.get_task(room.db, room.first)
    assert (task['status'], (task['result'] or {}).get('text')) == ('settled', 'Recovered from storage.')


def test_a_lost_run_that_could_not_be_resolved_clears_when_the_next_turn_needs_it(lost):
    room = lost()
    discard, failing = room.rpc.discard, [True]

    def timed_out_once(**kwargs):
        if failing:
            failing.clear()
            raise TimeoutError('transport call timed out after 30 seconds')
        return discard(**kwargs)

    room.rpc.discard = timed_out_once
    _pass_until(room.runtime, room.now, room.db, room.first, {'deferred'})
    assert _admissions(room.authority, room.sid) == ['unknown']
    later = _next_turn(room)
    _pass_until(room.runtime, room.now, room.db, later, {'running'})
    assert _admissions(room.authority, room.sid) == ['terminal', 'queued']


def test_a_store_left_with_a_cancelled_turns_lost_run_lets_the_next_turn_in(lost):
    """Stop used to clear only the row, so stores from before this fix can hold this state."""
    room = lost()
    lease = room.runtime._ensure_lease(room.runtime._rooms_provider()[0])
    state.recover_room(room.db, lease, clock=room.runtime.clock)
    state.defer_indeterminate_task(room.db, room.first, lease, expected_execution_generation=1,
                                   expected_cancel_generation=0, reason='member_unavailable', clock=room.runtime.clock)
    state.cancel_task(room.db, room.first, cancel_id='stop', expected_cancel_generation=0, clock=room.runtime.clock)
    later = _next_turn(room)
    _pass_until(room.runtime, room.now, room.db, later, {'running'})
    assert _admissions(room.authority, room.sid) == ['terminal', 'queued']


@pytest.fixture
def service_room(hosted_owner, tmp_path):
    """A lead's routing turn run by the gateway's own service, then a restart: a new service takes over."""
    authority, loop, _principal, _ = hosted_owner
    db, now = authority.db.db_path, [100.0]
    authority.profile_id = str(tmp_path)  # a real, served home
    homes = {'default': tmp_path, 'other': tmp_path}

    def service_at():
        service = CanonicalHostedRoomService(authority, loop)
        service.profile_homes = lambda: homes
        service.runtime.clock = lambda: now[0]
        service.runtime.lease_ttl_seconds, service.runtime.indeterminate_defer_seconds = 30, 5
        service.runtime._wake.wait = lambda _timeout: service.runtime._stop.set()
        return service

    first = service_at()
    first.authorize_room('durable-room-owner', ROOM, create=True)
    hosted_rooms.create_room(db, room_id=ROOM, name='Batch', members=MEMBERS,
                             authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    context.set_room_context(db, room_id=ROOM, event_id='ctx', purpose='Kernel work', notes='', routing='lead',
                             lead_member_id='one')
    first.send(room_id=ROOM, event_id='u1', payload={'text': 'Can someone look?', 'thread_id': 't1'})
    binding = first.bindings()[0]
    first.runtime._run_room_once(binding)
    routing, = state.list_tasks(db, room_id=ROOM, status='running')
    assert routing['payload']['lead_routing'] is True
    sid = next(iter(first.member_rpcs.values())).ref.session_id
    _restart(authority, sid)
    now[0] += 31  # the dead process's lease has lapsed
    service = service_at()
    return SimpleNamespace(db=db, now=now, service=service, runtime=service.runtime, authority=authority, sid=sid,
                           routing=routing['identity'],
                           turns=lambda: [t for t in state.list_tasks(db, room_id=ROOM)
                                          if t['identity'] != routing['identity']])


def _until_deferred_or_cancelled(room):
    _pass_until(room.runtime, room.now, room.db, room.routing, {'deferred', 'cancelled'})
    return state.get_task(room.db, room.routing)['status']


def _lead_answers(room):
    """The lead's own turn runs, for a message no retry or late answer routes a second time."""
    for _ in range(6):
        room.runtime._stop.clear()
        room.runtime._run_room_once(room.service.bindings()[0])
        if [t for t in room.turns() if t['status'] == 'running']:
            break
    turn, = room.turns()
    assert (turn['status'], turn['payload']['target_member_id']) == ('running', 'one'), room.runtime._last_error
    assert _admissions(room.authority, room.sid) == ['terminal', 'queued']


def test_on_the_real_service_a_slow_confirmation_still_gives_the_message_to_the_lead(service_room):
    room = service_room
    confirm = room.runtime.confirm_deferred_stop

    def slow(room_id, task):
        room.now[0] += 31  # longer than the lease the cancellation then commits under
        return confirm(room_id, task)

    room.runtime.confirm_deferred_stop = slow
    assert _until_deferred_or_cancelled(room) == 'cancelled'
    _lead_answers(room)
    events = hosted_rooms.read_events(room.db, room_id=ROOM, limit=hosted_rooms.MAX_LOG_LIMIT)['events']
    assert [e['payload']['reason'] for e in events if e['kind'] == 'turn.cancelled'] == ['lead_answers_itself']
    assert not [e for e in events if e['kind'] == 'room.assignment']


def _timed_out(*_args, **_kwargs):
    raise TimeoutError('transport call timed out after 30 seconds')


@pytest.fixture
def left_unknown(service_room, monkeypatch):
    """The turn was deferred with its lost run unresolved: that read timed out at the deadline."""
    room = service_room
    with monkeypatch.context() as patch:
        patch.setattr(HostedRoomAuthorityRPC, 'discard', _timed_out)
        assert _until_deferred_or_cancelled(room) == 'deferred'
    assert _admissions(room.authority, room.sid) == ['unknown']
    room.control = dict(member_id='one', task_id=room.routing.task_id, execution_generation=1)
    return room


@pytest.mark.parametrize('control', ['retry', 'stop', 'observe'])
def test_control_of_a_deferred_turn_resolves_its_lost_run(left_unknown, control):
    room = left_unknown
    if control == 'retry':
        assert room.service.retry_room_task(ROOM, **room.control)['status'] == 'queued'
        _pass_until(room.runtime, room.now, room.db, room.routing, {'running'})
    elif control == 'observe':
        _pass_until(room.runtime, room.now, room.db, room.routing, {'cancelled'}, limit=6)
        _lead_answers(room)
        assert state.get_task(room.db, room.routing)['execution_generation'] == 1
    else:
        assert room.service.stop_room(ROOM, cancel_id='stop') == 1
        assert state.get_task(room.db, room.routing)['status'] == 'cancelled'
    assert _admissions(room.authority, room.sid) == (['terminal'] if control == 'stop' else ['terminal', 'queued'])


def test_stop_keeps_a_deferred_turn_whose_lost_run_could_not_be_resolved(left_unknown, monkeypatch):
    room = left_unknown
    monkeypatch.setattr(HostedRoomAuthorityRPC, 'discard', _timed_out)
    assert room.service.stop_room(ROOM, cancel_id='stop') == 0
    assert state.get_task(room.db, room.routing)['status'] == 'deferred'
