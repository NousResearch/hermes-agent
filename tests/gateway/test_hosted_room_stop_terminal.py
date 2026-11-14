"""A local Stop request is not evidence that its canonical producer ended."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.session_hosted_service import CanonicalHostedRoomService
from tui_gateway import hosted_room_service
from hermes_state_runtime import claim_session_input, get_session_admission, settle_session_input


def _turn(hosted_owner, monkeypatch, *, started=True, admitted=True):
    """One planned group chat turn; ``started`` claims its canonical admission."""
    authority, loop, _, agent = hosted_owner
    authority.profile_id = str(Path(authority.db.db_path).parent)
    service = CanonicalHostedRoomService(authority, loop)
    monkeypatch.setattr(service, 'profile_homes', lambda: {
        'default': Path(authority.profile_id), 'other': Path(authority.profile_id) / 'other'})
    service.authorize_room('alice', 'room', create=True)
    rooms.create_room(authority.db.db_path, room_id='room', name='Room',
        authority_gateway_id=rooms.local_authority_gateway_id(),
        members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                 {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
    service.send(room_id='room', event_id='input', payload={'text': '@one input', 'thread_id': 'thread'})
    (queued,) = tasks.list_tasks(authority.db.db_path, room_id='room', status='queued')
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    attempt = tasks.start_task(authority.db.db_path, queued['identity'], lease, expected_cancel_generation=0,
                               clock=time.time)
    member = service._resolve_member_transport(binding, tasks.get_task(authority.db.db_path, attempt.identity))
    coords = dict(profile='default', source='bot_room')
    sid = member.create(**coords, title='Group: room')['session_id']
    receipt = member.submit(**coords, session_id=sid, prompt='input', task=attempt.identity,
        execution_generation=attempt.execution_generation,
        on_terminal=lambda result: service.runtime._on_terminal(binding, attempt, result)) if admitted else None
    if started and admitted:
        claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    turn = SimpleNamespace(authority=authority, service=service, binding=binding, lease=lease,
                           identity=attempt.identity, agent=agent, attempt=attempt, member=member, sid=sid)
    turn.task = lambda: tasks.get_task(authority.db.db_path, attempt.identity)
    turn.admission = lambda: get_session_admission(authority.db, admission_id=receipt['admission_id'])
    return turn


def _finish(turn, outcome, text=''):
    """The bot's turn commits its outcome."""
    canonical = turn.admission()
    settle_session_input(turn.authority.db, epoch=turn.authority.epoch, admission_id=canonical['admission_id'],
        generation=canonical['generation'], outcome=outcome,
        result={'result': {'interrupted': outcome == 'interrupted', 'final_response': text}, 'usage': {}})


@pytest.mark.parametrize('producer_state,outcome', [
    ('queued', 'interrupted'), ('started', 'interrupted'), ('started', 'completed'), ('deadline', 'interrupted')])
def test_room_stop_waits_for_the_producer_terminal(hosted_owner, monkeypatch, producer_state, outcome):
    turn = _turn(hosted_owner, monkeypatch, started=producer_state != 'queued')
    if producer_state == 'deadline':
        tasks.begin_task_cancel(turn.service.db_path, turn.identity, cancel_id='deadline:1',
                                expected_cancel_generation=0, clock=time.time)
    turn.service.stop_room('room', cancel_id='stop')  # what groups.stop runs
    if producer_state == 'queued':
        # Keeps working: a turn that never started is stopped at once.
        assert turn.task()['status'] == 'cancelled'
        assert (turn.admission()['status'], turn.admission()['outcome']) == ('terminal', 'cancelled')
        assert not turn.agent.interrupted
        return
    assert turn.agent.interrupted and turn.admission()['status'] == 'started'
    assert turn.task()['status'] == 'stopping', {
        'hosted_status': turn.task()['status'], 'canonical_status': turn.admission()['status']}
    turn.service.send(room_id='room', event_id='next', payload={'text': '@two next', 'thread_id': 'next'})
    assert tasks.list_tasks(turn.service.db_path, room_id='room', status='queued') == []
    _finish(turn, outcome, 'the answer' if outcome == 'completed' else '')
    # This read can publish the reply on the owner loop, under the policy lock.
    turn.service.stop_room('room', cancel_id='stop')
    expected = 'failed' if producer_state == 'deadline' else 'settled' if outcome == 'completed' else 'cancelled'
    assert turn.task()['status'] == expected
    if expected == 'failed':
        assert turn.task()['result']['reason_code'] == 'turn_deadline_exceeded'
    texts = [e['payload'].get('text') for e in turn.service._events('room')]
    assert texts.count('the answer') == int(outcome == 'completed')
    assert turn.service.runtime.status()['last_error'] is None
    turn.service.prepare_room(turn.binding)
    next_task, = tasks.list_tasks(turn.service.db_path, room_id='room', status='queued')
    tasks.start_task(turn.service.db_path, next_task['identity'], turn.lease,
                     expected_cancel_generation=0, clock=time.time)


@pytest.mark.parametrize('deferred', [False, True])
def test_disband_waits_for_a_local_turn_to_wind_down(hosted_owner, monkeypatch, deferred):
    from gateway.session_group_controls import _group
    turn = _turn(hosted_owner, monkeypatch)
    if deferred:
        tasks.mark_attempt_indeterminate(turn.service.db_path, turn.attempt, turn.lease, clock=time.time)
        tasks.defer_indeterminate_task(turn.service.db_path, turn.identity, turn.lease,
            expected_execution_generation=1, expected_cancel_generation=0, reason='member_unavailable', clock=time.time)
    turn.authority.hosted_room_service = turn.service
    monkeypatch.setattr(turn.service.runtime, 'status', lambda: {'running': True, 'stopping': False})
    def disband():
        return _group(turn.authority, SimpleNamespace(subject='alice'), 'groups.disband',
                      {'room_id': 'room', 'cancel_id': 'end'})
    monkeypatch.setattr(hosted_room_service, '_DISBAND_STOP_WAIT_SECONDS', 0)
    with pytest.raises(rooms.RoomStopPendingError):
        disband()
    assert turn.agent.interrupted and turn.task()['status'] == ('deferred' if deferred else 'stopping')
    assert rooms.room_state(turn.service.db_path, room_id='room').get('disbanded_at') is None
    _finish(turn, 'interrupted')
    ended = disband()
    assert ended['tombstone']['room_id'] == 'room'
    assert turn.agent.interrupted and turn.task()['status'] == 'cancelled'


def test_disband_needs_one_press_when_the_bot_stops_at_once(hosted_owner, monkeypatch):
    """Disband, with the bot ending and the worker's pass running just after Stop reaches it."""
    turn = _turn(hosted_owner, monkeypatch)
    from gateway.session_group_controls import _group
    turn.authority.hosted_room_service = turn.service
    runtime = turn.service.runtime
    monkeypatch.setattr(runtime, 'status', lambda: {'running': True, 'stopping': False})
    monkeypatch.setattr(hosted_room_service, '_DISBAND_STOP_WAIT_SECONDS', 10)
    original = runtime.cancel

    def wind_down():
        _finish(turn, 'interrupted')
        runtime._finish_stop(turn.binding, turn.task(), runtime._ensure_lease(turn.binding))

    def cancel_then_wind_down(*args, **kwargs):
        result = original(*args, **kwargs)
        threading.Timer(0.2, wind_down).start()
        return result

    monkeypatch.setattr(runtime, 'cancel', cancel_then_wind_down)
    ended = _group(turn.authority, SimpleNamespace(subject='alice'), 'groups.disband',
                   {'room_id': 'room', 'cancel_id': 'end'})
    assert ended['tombstone']['room_id'] == 'room'
    assert turn.task()['status'] == 'cancelled'


def test_disband_counts_work_a_message_started_while_it_waited(hosted_owner, monkeypatch):
    turn = _turn(hosted_owner, monkeypatch)
    turn.service.stop_room('room', cancel_id='stop')
    _finish(turn, 'interrupted')
    turn.service.runtime._finish_stop(turn.binding, turn.task(), turn.service.runtime._ensure_lease(turn.binding))
    assert turn.task()['status'] == 'cancelled'
    monkeypatch.setattr(hosted_room_service, '_DISBAND_STOP_WAIT_SECONDS', 0)
    assert turn.service._stops_still_pending('room') == 0
    turn.service.send(room_id='room', event_id='next', payload={'text': '@two next', 'thread_id': 'next'})
    turn.service.prepare_room(turn.binding)
    assert tasks.list_tasks(turn.service.db_path, room_id='room', status='queued')
    assert turn.service._stops_still_pending('room') == 1


@pytest.mark.parametrize('stop', ['user', 'deadline'])
@pytest.mark.parametrize('outcome', ['interrupted', 'completed'])
def test_a_local_bot_that_never_finishes_releases_the_room_at_its_limit(hosted_owner, monkeypatch, stop, outcome):
    turn = _turn(hosted_owner, monkeypatch)
    runtime = turn.service.runtime
    now = [turn.task()['started_at']]
    runtime.clock = lambda: now[0]
    runtime.turn_timeout_seconds, runtime.indeterminate_defer_seconds = 10, 5
    if stop == 'user':
        turn.service.stop_room('room', cancel_id='stop')
    else:
        now[0] += 10
        runtime._expire_attempt_deadline(turn.binding, turn.task(), turn.lease)
    assert turn.task()['status'] == 'stopping' and turn.admission()['status'] == 'started'
    now[0] += 6 if stop == 'user' else 4
    assert runtime._retry_stopping_tasks(turn.binding, turn.lease) is True
    now[0] = turn.task()['started_at'] + 16
    # Exercise the active worker too: its stopping branch must not bypass the bound.
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(runtime._wait_for_terminal, turn.binding, profile='default',
            session_id=turn.sid, attempt=turn.attempt, transport=turn.member, deadline_monotonic=0)
        try:
            assert waiting.result(timeout=10) is None
        finally:
            runtime._stop.set()
            runtime.wakeup()
    assert turn.task()['status'] == 'deferred' and turn.admission()['status'] == 'started'
    with pytest.raises(rooms.RoomStopPendingError):
        turn.service.stop_room('room', cancel_id='end', require_acknowledged=True)
    turn.service.send(room_id='room', event_id='next', payload={'text': '@two next', 'thread_id': 'next'})
    next_task, = tasks.list_tasks(turn.service.db_path, room_id='room', status='queued')
    tasks.start_task(turn.service.db_path, next_task['identity'], runtime._ensure_lease(turn.binding),
                     expected_cancel_generation=0, clock=runtime.clock)
    _finish(turn, outcome, 'late answer')
    result = runtime.cancel(turn.identity, cancel_id='confirmed')
    expected = 'settled' if outcome == 'completed' else 'failed' if stop == 'deadline' else 'cancelled'
    assert result['status'] == expected
    if expected == 'failed':
        assert result['result']['reason_code'] == 'turn_deadline_exceeded'
    if expected == 'settled':
        assert [e['payload']['text'] for e in turn.service._events('room')
                if e['kind'] == 'message.member'] == ['late answer']
