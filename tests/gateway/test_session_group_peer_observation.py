"""A member lost after lease renewal cannot hold the group chat indefinitely."""

import asyncio
import time

import pytest

from gateway import hosted_room_driver as driver
from gateway.platforms import api_server_runs
from tests.gateway import test_session_group_peers as peers
from tests.tui_gateway.test_hosted_room_driver_runtime import FakeSessionRPC
from hermes_agent.tui_gateway.hosted_room_driver import HostedRoomRuntime

gateway, linked_peer = peers.gateway, peers.linked_peer


@pytest.mark.asyncio
@pytest.mark.parametrize('resolution', ['retry', 'discard', 'transient'])
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_renewed_peer_turn_recovers_after_member_goes_away(
        gateway, monkeypatch, linked_peer, resolution):
    server, url, catalog, room = linked_peer
    assert (await peers._seat(gateway, room, url, catalog))[1]['registered']
    service, db = gateway.service, gateway.db.db_path
    runtime, binding = service.runtime, service.bindings()[0]
    runtime.status = lambda: {**HostedRoomRuntime.status(runtime), 'running': True, 'stopping': False}
    now = [time.time()]
    start = now[0]
    runtime.clock = lambda: now[0]
    runtime.active_poll_interval_seconds = 0
    local = FakeSessionRPC()
    resolve = runtime.transport_resolver
    runtime.transport_resolver = lambda b, t: (
        resolve(b, t) if t['payload'].get('target_member_id') == 'reviewer' else local)

    async def running(adapter, run, **kwargs):
        adapter._set_run_status(run.run_id, 'running')

    monkeypatch.setattr(api_server_runs, '_execute_run', running)
    client = service.peer_clients[('linked', 'reviewer')]
    client.clock = runtime.clock
    history, reads = client.history, []
    loop = asyncio.get_running_loop()
    port = server.port

    def lose_member_after_renewal(**kwargs):
        reads.append(now[0])
        if len(reads) == 2:
            now[0] += 16
            asyncio.run_coroutine_threadsafe(server.close(), loop).result(10)
        result = history(**kwargs)
        if len(reads) == 1:
            now[0] += 16
        return result

    monkeypatch.setattr(client, 'history', lose_member_after_renewal)
    assert (await peers.call(gateway.owner, 'groups.send', room_id='linked', event_id='ask',
                            payload={'text': '@reviewer review', 'thread_id': 'peer-thread'}))['accepted']
    task, = driver.list_tasks(db, room_id='linked', status='queued')
    identity = task['identity']
    restarted = None
    lost = start + 33

    def tick(_timeout):
        nonlocal restarted
        assert now[0] < start + 200, 'the turn was never retired'
        now[0] += 1
        if resolution == 'transient' and now[0] == lost + 10:
            restarted, _ = asyncio.run_coroutine_threadsafe(
                peers.serve(gateway, monkeypatch, port=port, stop=True), loop).result(10)
            gateway.adapter._set_run_status(
                next(iter(gateway.adapter._run_statuses)), 'completed', output='Recovered reply', completed=True)

    monkeypatch.setattr(runtime._wake, 'wait', tick)
    try:
        await asyncio.wait_for(asyncio.to_thread(runtime._process_room, binding), 30)
        assert reads[:2] == [start, start + 17]
        assert 'observation failed after submit' in runtime._last_error
        run_id, = gateway.adapter._run_statuses
        if resolution == 'transient':
            # Back within the grace period: the watcher never left, so the same run settles.
            assert driver.get_task(db, identity)['status'] == 'settled'
            assert driver.get_task(db, identity)['result']['text'] == 'Recovered reply'
        else:
            assert driver.get_task(db, identity)['status'] == 'indeterminate'
            assert now[0] - lost <= runtime.lease_ttl_seconds + 2
            assert await peers.linked_room(gateway, catalog, room_id='other')
            assert (await peers.call(gateway.owner, 'groups.send', room_id='other', event_id='other-ask',
                                    payload={'text': '@host answer', 'thread_id': 'other'}))['accepted']
            other = next(b for b in service.bindings() if b.room_id == 'other')
            await asyncio.to_thread(runtime._process_room, other)
            assert driver.list_tasks(db, room_id='other')[0]['status'] == 'settled'
            now[0] += runtime.indeterminate_defer_seconds
            await asyncio.to_thread(runtime._process_room, binding)
            assert driver.get_task(db, identity)['status'] == 'deferred'
            assert now[0] - lost <= runtime.lease_ttl_seconds + runtime.indeterminate_defer_seconds + 2
            actions = service.status('linked')['pending_actions']
            assert {a['kind'] for a in actions if a.get('task_id') == identity.task_id} == {'retry', 'discard'}
            assert (await peers.call(gateway.owner, 'groups.send', room_id='linked', event_id='next',
                payload={'text': '@host next', 'thread_id': 'next-thread'}))['accepted']
            await asyncio.to_thread(runtime._process_room, binding)
            assert any(t['status'] == 'settled' and t['payload']['target_member_id'] == 'host'
                       for t in driver.list_tasks(db, room_id='linked'))
            restarted, _ = await peers.serve(gateway, monkeypatch, port=port, stop=True)
            gateway.adapter._set_run_status(run_id, 'completed', output='Recovered reply', completed=True)
            now[0] += client.poll_max_seconds + 1
            result = await peers.call(gateway.owner, 'groups.' + resolution, room_id='linked',
                member_id='reviewer', task_id=identity.task_id, execution_generation=1)
            assert isinstance(result, dict), result
            assert driver.get_task(db, identity)['status'] == ('settled' if resolution == 'retry' else 'cancelled')
        assert driver.get_task(db, identity)['execution_generation'] == 1
        assert list(gateway.adapter._run_statuses) == [run_id]
        await asyncio.to_thread(runtime._process_room, binding)  # publish any recovered terminal row
        replies = [e for e in service._events('linked') if e['kind'] == 'message.member'
                   and e['payload'].get('text') == 'Recovered reply']
        assert len(replies) == (0 if resolution == 'discard' else 1)
    finally:
        if restarted is not None:
            await restarted.close()
