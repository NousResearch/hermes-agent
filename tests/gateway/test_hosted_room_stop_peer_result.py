"""A confirmed peer Stop still publishes a reply that finished during deferral."""
import asyncio

import pytest

from gateway import hosted_room_driver as tasks
from tests.gateway import test_session_group_peers as peers

gateway, linked_peer = peers.gateway, peers.linked_peer


@pytest.mark.asyncio
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_deferred_peer_stop_keeps_the_finished_reply(gateway, monkeypatch, linked_peer):
    service, db, identity = await peers._deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    binding = service.bindings()[0]
    task = tasks.get_task(db, identity)
    assert not await asyncio.to_thread(service.close_peer_admission, binding, task)
    run_id, = gateway.adapter._run_statuses
    gateway.adapter._set_run_status(run_id, 'completed', output='finished while stopping')
    await asyncio.to_thread(service.stop_room, 'linked', cancel_id='stop')
    saved = tasks.get_task(db, identity)
    assert saved['status'] == 'settled'
    assert saved['result']['text'] == 'finished while stopping'
    assert [e['payload']['text'] for e in service._events('linked')
            if e['kind'] == 'message.member'] == ['finished while stopping']


def test_delayed_peer_deadline_acknowledgement_keeps_deadline_cause(tmp_path):
    from tests.tui_gateway.test_hosted_room_driver_runtime import (
        BINDING, FakeSessionRPC, _admit, _identity, _runtime,
    )
    from gateway import hosted_rooms
    db = tmp_path / 'peer-deadline.db'
    hosted_rooms.create_room(db, room_id=BINDING.room_id, name='Deadline', members=[],
                             authority_gateway_id=BINDING.gateway_id)
    identity, now = _identity(), [100.0]
    _admit(db, identity)
    peer = FakeSessionRPC(auto_complete=False)
    peer.add_session(active=True, task_id=identity.task_id)
    peer.interrupt = lambda **kw: {'interrupted': False}
    runtime = _runtime(db, None, transport_resolver=lambda *_: peer, clock=lambda: now[0],
                       remote_member=lambda *_: True, lease_ttl_seconds=100, turn_timeout_seconds=5)
    lease = runtime._ensure_lease(BINDING)
    tasks.start_task(db, identity, lease, expected_cancel_generation=0, clock=runtime.clock)
    runtime._expire_attempt_deadline(BINDING, tasks.get_task(db, identity), lease)
    assert tasks.get_task(db, identity)['status'] == 'stopping'
    now[0] += 1
    peer.info = lambda **kw: {'active': False, 'status': 'cancelled',
                             'task_id': identity.task_id, 'execution_generation': 1}
    assert runtime._retry_stopping_tasks(BINDING, lease) is False
    assert tasks.get_task(db, identity)['result']['reason_code'] == 'turn_deadline_exceeded'
