"""Stop and Discard must describe what happened to the selected attempt."""

import pytest

from gateway import hosted_room_driver as state, hosted_rooms
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_hosted_result_polling import deferred_turn as deferred_turn_fixture, _finish

deferred_turn = deferred_turn_fixture


def test_stop_of_a_deferred_local_turn_reaches_the_running_bot(deferred_turn):
    room = deferred_turn
    assert room.rpc.info(**room.coords)['active'] is True
    assert room.service.stop_room('room', cancel_id='stop') == 1
    assert room.agent.interrupted
    assert state.get_task(room.db, room.identity)['status'] == 'cancelled'


def test_local_deferred_turn_offers_retry_and_rejects_discard(deferred_turn):
    room = deferred_turn
    with pytest.raises(RuntimeStoreError, match='stale_generation'):
        room.service.discard_room_task('room', **room.control)
    assert {a['kind'] for a in room.service.status('room')['pending_actions']} == {'retry'}
    assert state.get_task(room.db, room.identity)['status'] == 'deferred'


def test_a_transient_submit_authorization_change_keeps_retry(deferred_turn, monkeypatch):
    room = deferred_turn
    room.runtime.prepare_room = room.runtime.publish_terminal = None
    _finish(room, room.row, 'Earlier work.')
    room.service.retry_room_task('room', **room.control)
    later = state.TaskIdentity('room', 'later', 'thread', 'later')
    state.admit_task(room.db, later, payload={'target_profile': 'default', 'target_member_id': 'one',
        'source_event_seq': 1, 'prompt': 'valid request'}, clock=room.runtime.clock)
    submit = room.rpc.submit

    def home_changed(**kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(room.service, 'profile_homes', dict)
            return submit(**kwargs)

    monkeypatch.setattr(room.rpc, 'submit', home_changed)
    room.runtime._stop.clear()
    room.runtime._run_room_once(room.binding)
    # Main recovers ambiguous submits after the keeper's last renewed lease expires.
    room.now[0] += 31
    room.runtime._run_room_once(room.binding)
    assert state.get_task(room.db, later)['status'] == 'indeterminate'
    monkeypatch.setattr(room.rpc, 'submit', submit)
    room.now[0] += 61
    room.runtime._run_room_once(room.binding)
    assert state.get_task(room.db, later)['status'] == 'deferred'
    assert room.service.retry_room_task('room', member_id='one', task_id='later',
                                        execution_generation=1)['status'] == 'queued'
    room.runtime._run_room_once(room.binding)
    assert state.get_task(room.db, later)['status'] == 'running'


@pytest.mark.parametrize('control', ['retry', 'stop'])
def test_completion_between_receipt_and_idle_read_is_published(deferred_turn, monkeypatch, control):
    room = deferred_turn
    info, answer = room.rpc.info, 'Finished during the control probe.'

    def finish_before_idle(**kwargs):
        _finish(room, room.row, answer)
        monkeypatch.setattr(room.rpc, 'info', info)
        return info(**kwargs)

    monkeypatch.setattr(room.rpc, 'info', finish_before_idle)
    if control == 'retry':
        room.service.retry_room_task('room', **room.control)
    else:
        room.service.stop_room('room', cancel_id='stop')
    task = state.get_task(room.db, room.identity)
    assert (task['status'], (task['result'] or {}).get('text')) == ('settled', answer)
    assert task['execution_generation'] == 1
    events = hosted_rooms.read_events(room.db, room_id='room')['events']
    assert [e['payload']['text'] for e in events if e['kind'] == 'message.member'] == [answer]
