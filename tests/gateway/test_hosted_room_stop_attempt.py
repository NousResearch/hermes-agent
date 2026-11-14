"""Stop follows current work while delayed worker delivery retains its attempt."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from gateway import hosted_room_driver as tasks
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state_runtime import claim_session_input
from tests.gateway.test_hosted_room_stop_terminal import _turn, _finish


def _defer(turn):
    stopping = tasks.begin_task_cancel(turn.service.db_path, turn.identity, cancel_id='stop',
                                      expected_cancel_generation=0, clock=time.time)
    deferred = tasks.defer_unconfirmed_stop(turn.service.db_path, turn.identity, turn.lease,
        expected_execution_generation=1, expected_cancel_generation=stopping['cancel_generation'],
        reason='member_unavailable', clock=time.time)
    turn.service.runtime._publish(turn.binding, deferred)
    return deferred


def _legacy_retry(turn, monkeypatch):
    """An older producer has no exact lookup and retains its existing Retry behavior."""
    turn.member.callbacks.clear()  # the older producer has no push receipt; only its idle probe arrives
    original = turn.member._info

    async def info(params):
        result = await original(params)
        result.pop('lookup_attempt', None)
        return result

    with monkeypatch.context() as legacy:
        legacy.setattr(turn.member, '_info', info)
        return turn.service.retry_room_task('room', member_id='one', task_id=turn.identity.task_id,
                                            execution_generation=1)


@pytest.mark.parametrize(('restart', 'supported'), [(False, True), (True, True), (False, False)])
def test_stop_delivery_is_bounded_and_fenced(hosted_owner, monkeypatch, restart, supported):
    turn = _turn(hosted_owner, monkeypatch)
    calls = []
    monkeypatch.setattr(turn.agent, 'interrupt', lambda: calls.append('interrupt'))
    original = turn.member._info

    async def old_info(params):
        result = await original(params)
        result.pop('interrupt_execution_generation', None)
        return result

    if not supported:
        monkeypatch.setattr(turn.member, '_info', old_info)
    turn.service.stop_room('room', cancel_id='stop')
    if restart:
        homes = turn.service.profile_homes()
        tasks.release_lease(turn.service.db_path, turn.lease, clock=time.time)
        turn.service = CanonicalHostedRoomService(turn.authority, turn.member.loop)
        monkeypatch.setattr(turn.service, 'profile_homes', lambda: homes)
        turn.lease = turn.service.runtime._ensure_lease(turn.binding)
    for _ in range(3):
        assert turn.service.runtime._retry_stopping_tasks(turn.binding, turn.lease)
    assert calls == ['interrupt'] * ((2 if restart else 1) if supported else 0)
    assert turn.task()['status'] == 'stopping'
    _finish(turn, 'completed', 'finished answer')
    turn.service.stop_room('room', cancel_id='stop')
    assert turn.task()['status'] == 'settled'


def test_delayed_deferred_stop_cannot_interrupt_or_cancel_a_retry(hosted_owner, monkeypatch):
    turn = _turn(hosted_owner, monkeypatch)
    _defer(turn)
    entered, release = threading.Event(), threading.Event()
    original = turn.member._call

    def delayed(operation, **params):
        if operation == 'interrupt':
            entered.set()
            assert release.wait(10)
        return original(operation, **params)

    monkeypatch.setattr(turn.member, '_call', delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        stopped = executor.submit(turn.service.runtime.cancel, turn.identity, cancel_id='late-stop')
        try:
            assert entered.wait(5)
            _finish(turn, 'interrupted')
            queued = _legacy_retry(turn, monkeypatch)
            attempt = tasks.start_task(turn.service.db_path, turn.identity, turn.lease,
                expected_cancel_generation=queued['cancel_generation'], clock=time.time)
            turn.member.submit(profile='default', source='bot_room', session_id=turn.sid,
                prompt='retry', task=turn.identity, execution_generation=attempt.execution_generation,
                on_terminal=None)
            claim_session_input(turn.authority.db, epoch=turn.authority.epoch, session_id=turn.sid)
        finally:
            release.set()
        stopped.result(timeout=10)
    assert attempt.execution_generation == 2
    assert not turn.agent.interrupted, 'generation 1 Stop reached generation 2'
    assert turn.task()['status'] == 'running'


def test_deferred_stop_accepts_its_late_terminal_callback(hosted_owner, monkeypatch):
    turn = _turn(hosted_owner, monkeypatch)
    _defer(turn)
    _finish(turn, 'completed', 'late callback')
    turn.service.runtime._on_terminal(turn.binding, turn.attempt, {'status': 'settled', 'text': 'late callback'})
    assert turn.task()['status'] == 'settled'
    assert [e['payload']['text'] for e in turn.service._events('room')
            if e['kind'] == 'message.member'] == ['late callback']


@pytest.mark.parametrize('started_again', [False, True])
def test_stop_cancels_a_running_attempt_requeued_after_non_admission(hosted_owner, monkeypatch, started_again):
    turn = _turn(hosted_owner, monkeypatch, admitted=False)
    cancel = turn.service.runtime.cancel

    def refused_before_cancel(identity, **kwargs):
        queued = tasks.requeue_not_admitted_task(turn.service.db_path, turn.attempt, clock=time.time)
        if started_again:
            attempt = tasks.start_task(turn.service.db_path, identity, turn.lease,
                expected_cancel_generation=queued['cancel_generation'], clock=time.time)
            turn.member.submit(profile='default', source='bot_room', session_id=turn.sid, prompt='retry',
                task=identity, execution_generation=attempt.execution_generation, on_terminal=None)
            claim_session_input(turn.authority.db, epoch=turn.authority.epoch, session_id=turn.sid)
        return cancel(identity, **kwargs)

    monkeypatch.setattr(turn.service.runtime, 'cancel', refused_before_cancel)
    assert turn.service.stop_room('room', cancel_id='stop-refused') == 1
    assert turn.task()['status'] == ('stopping' if started_again else 'cancelled')
    if started_again:
        assert turn.task()['execution_generation'] == 2 and turn.agent.interrupted
    turn.service.prepare_room(turn.binding)
    assert tasks.list_tasks(turn.service.db_path, room_id='room', status='queued') == []
