"""Stop confirmation and racing answers over main's private-owner socket fixture."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from hermes_state_runtime import settle_session_input
from tests.gateway.test_hosted_room_attempt_home_moves import homes as homes_fixture

homes = homes_fixture


def _started(homes):
    rpc = homes.submit(homes.a)
    return rpc, homes.claim(homes.a)


def _finish(homes, row, outcome, text=''):
    authority = homes.targets[homes.a]
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
        generation=row['generation'], outcome=outcome,
        result={'result': {'final_response': text}, 'usage': {}})


@pytest.mark.parametrize('deferred', [False, True])
def test_disband_keeps_a_bot_until_its_owner_confirms_terminal(homes, deferred):
    _, row = _started(homes)
    if deferred:
        homes.defer(homes.a)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(homes.service.stop_room, 'room', cancel_id='stop').result(10) == (0 if deferred else 1)
    assert homes.task['status'] == ('deferred' if deferred else 'stopping')
    lock = homes.service._room_approval_stop_lock('room')
    assert lock.acquire(timeout=5)
    lock.release()
    with pytest.raises(rooms.RoomStopPendingError):
        homes.disband()
    assert homes.agent.interrupted and homes.rows(homes.a)[0]['status'] == 'started'
    assert homes.service._room('room').get('disbanded_at') is None
    _finish(homes, row, 'interrupted')
    assert homes.disband()['tombstone']['room_id'] == 'room'
    assert homes.task['status'] == 'cancelled'


@pytest.mark.parametrize('restart', [False, True])
def test_reply_committing_between_history_and_idle_is_retained(homes, monkeypatch, restart):
    _, row = _started(homes)
    tasks.begin_task_cancel(homes.service.db_path, homes.source.identity, cancel_id='stop',
        expected_cancel_generation=0, clock=homes.service.runtime.clock)
    if restart:
        homes.restart()
    published, read = [], []
    runtime = homes.service.runtime
    runtime.publish_terminal = lambda binding, task: published.append(task)
    info, history = HostedRoomAuthorityRPC._info, HostedRoomAuthorityRPC._history

    async def read_before_commit(rpc, params):
        result = await history(rpc, params)
        read.append(True)
        return result

    async def commit_before_idle(rpc, params):
        if read and homes.rows(homes.a)[0]['status'] == 'started':
            _finish(homes, row, 'completed', 'answer in the gap')
        result = await info(rpc, params)
        result.pop('lookup_attempt', None)  # older-daemon history fallback
        return result

    monkeypatch.setattr(HostedRoomAuthorityRPC, '_info', commit_before_idle)
    monkeypatch.setattr(HostedRoomAuthorityRPC, '_history', read_before_commit)
    homes.service.stop_room('room', cancel_id='stop')
    assert homes.task['status'] == 'settled'
    assert [t['result']['text'] for t in published] == ['answer in the gap']
