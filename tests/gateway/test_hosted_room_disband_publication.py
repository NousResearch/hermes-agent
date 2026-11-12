"""Disband fences publication without holding service locks across peer recovery."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from hermes_agent.gateway import hosted_room_driver as driver, hosted_room_links, hosted_rooms
from tests.gateway.test_session_group_peers import (
    gateway as gateway_fixture, linked_peer as linked_peer_fixture, linked_room, _seat, call, invite, _owed, _grant_error)
from hermes_agent.tui_gateway import methods_groups, methods_groups_forward, server
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient

gateway = gateway_fixture
linked_peer = linked_peer_fixture


def _surface(gateway, monkeypatch, surface):
    if surface == 'canonical':
        return lambda method, **params: call(gateway.owner, method, **params)
    monkeypatch.setattr(methods_groups, '_service', gateway.service)
    monkeypatch.setattr(methods_groups_forward, '_discover', lambda home: SimpleNamespace(state='absent'))

    async def request(method, **params):
        reply = await asyncio.to_thread(server.handle_request, {'id': 1, 'method': method, 'params': params})
        return reply.get('result', reply.get('error'))
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_registration_demoted_after_probe_persists_revoke_debt(gateway, monkeypatch, linked_peer, surface):
    from hermes_agent.gateway.hosted_room_replicas import demote_room
    from hermes_agent.gateway.session_hosted_service import CanonicalHostedRoomService

    _peer, url, catalog, room = linked_peer
    service, probe = gateway.service, PeerRunsHTTPClient.probe
    request = _surface(gateway, monkeypatch, surface)
    grant = (await invite(gateway, room))['grant']

    def demote_after_probe(client, **kwargs):
        result = probe(client, **kwargs)
        demote_room(service.db_path, room_id='linked', observed_gateway_id='install:new-owner',
                    observed_epoch=room['authority_epoch'] + 1)
        return result

    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', demote_after_probe)
    await request('groups.peer.register', room_id='linked', member_id='reviewer',
                  target_url=url, target_profile='default', grant=grant, catalog=catalog)
    monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe)
    with pytest.raises(hosted_rooms.AuthorityConflictError):
        service.route_authority('linked')
    assert service.peer_routes == {} and hosted_room_links.load_room_links(service.db_path) == ()
    assert await _grant_error(url, grant) is None
    assert _owed(service.db_path) == [('linked', 'reviewer', 0)]
    restarted = CanonicalHostedRoomService(gateway.authority, None)
    assert await asyncio.to_thread(restarted.retry_owed_revokes) == 1
    assert _owed(service.db_path) == []
    assert await _grant_error(url, grant) == 'room_reauthorization_required'


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('operation', ['disband', 'stop'])
async def test_expiring_active_peer_can_stop_without_holding_publication_locks(gateway, monkeypatch, linked_peer, surface, operation):
    from hermes_agent.gateway.platforms import api_server_runs

    async def complete(adapter, run, **kwargs):
        adapter._set_run_status(run.run_id, 'completed', output='peer reply', completed=True)

    monkeypatch.setattr(api_server_runs, '_execute_run', complete)
    _peer, url, catalog, room = linked_peer
    service, blocked = gateway.service, []
    lock = threading.Lock()

    class BoundedLock:
        def __enter__(self):
            if not lock.acquire(timeout=2):
                blocked.append('route')
                raise RuntimeError('publication lock blocked recovery')

        def __exit__(self, *args):
            lock.release()

    refresh = PeerRunsHTTPClient.refresh_grant

    def refresh_without_policy_lock(client, **kwargs):
        def check_policy():
            if service._policy_lock.acquire(timeout=2):
                service._policy_lock.release()
            else:
                blocked.append('policy')
        checker = threading.Thread(target=check_policy)
        checker.start()
        checker.join(5)
        assert not checker.is_alive()
        return refresh(client, **kwargs)

    request = _surface(gateway, monkeypatch, surface)
    _, registered = await _seat(gateway, room, url, catalog, ttl_seconds=120, status_ttl_seconds=3600)
    assert registered['registered']
    other = await linked_room(gateway, catalog, room_id='other')
    assert (await call(gateway.owner, 'groups.send', room_id='linked', event_id='ask',
                       payload={'text': '@reviewer review', 'thread_id': 'thread'}))['accepted']
    task, = driver.list_tasks(service.db_path, room_id='linked', status='queued')
    binding = next(b for b in service.bindings() if b.room_id == 'linked')
    lease = service.runtime._ensure_lease(binding)
    driver.start_task(service.db_path, task['identity'], lease, expected_cancel_generation=0, clock=service.runtime.clock)
    monkeypatch.setattr(service, 'peer_route_lock', BoundedLock())
    monkeypatch.setattr(PeerRunsHTTPClient, 'refresh_grant', refresh_without_policy_lock)
    try:
        result = await request(f'groups.{operation}', room_id='linked')
    except RuntimeError as exc:  # the canonical surface raises Stop's refusal to its transport
        result = {'message': str(exc)}
    assert not blocked, blocked
    if operation == 'disband':
        # Disband waits for the member gateway's Stop: this attempt never reached it, so its Stop
        # cannot be confirmed and the room and its route stay for a retry.
        assert ('room_stop_pending' if surface == 'canonical' else 'still stopping') in str(result), result
        assert ('linked', 'reviewer') in service.peer_routes
        assert len(hosted_room_links.load_room_links(service.db_path)) == 1
    else:
        assert isinstance(result, dict) and 'code' not in result, result
    assert (await _seat(gateway, other, url, catalog))[1]['registered']
    assert (await call(gateway.owner, 'groups.send', room_id='other', event_id='next',
                       payload={'text': '@reviewer next', 'thread_id': 'next'}))['accepted']
    binding = next(b for b in service.bindings() if b.room_id == 'other')
    await asyncio.to_thread(service.runtime._run_room_once, binding)
    task, = driver.list_tasks(service.db_path, room_id='other', status='settled')
    assert task['result']['text'] == 'peer reply'


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('phase', ['before_probe', 'after_probe'])
@pytest.mark.parametrize('outcome', ['failed', 'ended', 'ended_without_route'])
async def test_registration_during_disband_preserves_surviving_grants(gateway, monkeypatch, linked_peer, surface, phase, outcome):
    from hermes_agent.gateway.platforms import api_server_runs
    from hermes_agent.hermes_state_runtime import RuntimeStoreError
    from hermes_agent.tui_gateway.hosted_room_peer_transport import build_member_dispatch

    async def complete(adapter, run, **kwargs):
        adapter._set_run_status(run.run_id, 'completed', output='peer reply', completed=True)

    monkeypatch.setattr(api_server_runs, '_execute_run', complete)
    _peer, url, catalog, room = linked_peer
    reached, release, probed, publish = (threading.Event() for _ in range(4))
    service, stop, key = gateway.service, gateway.service.stop_room, ('linked', 'reviewer')
    ending = seating = None

    def paused_stop(*args, **kwargs):
        reached.set()
        assert release.wait(10)
        return stop(*args, **kwargs)

    async def disband(request):
        try:
            return await request('groups.disband', room_id='linked')
        except RuntimeError as exc:
            return {'message': str(exc)}

    try:
        request = _surface(gateway, monkeypatch, surface)
        if outcome == 'ended_without_route':
            grant = (await invite(gateway, room))['grant']
        else:
            grant, registered = await _seat(gateway, room, url, catalog, ttl_seconds=120, status_ttl_seconds=3600)
            assert registered['registered']
        routes, links = dict(service.peer_routes), hosted_room_links.load_room_links(service.db_path)
        binding = next(b for b in service.bindings() if b.room_id == 'linked')
        if outcome == 'failed':
            assert (await call(gateway.owner, 'groups.send', room_id='linked', event_id='local',
                               payload={'text': '@host wait', 'thread_id': 'local'}))['accepted']
            task, = driver.list_tasks(service.db_path, room_id='linked', status='queued')
            lease = service.runtime._ensure_lease(binding)
            driver.start_task(service.db_path, task['identity'], lease, expected_cancel_generation=0,
                              clock=service.runtime.clock)
            monkeypatch.setattr(service.runtime, 'cancel', lambda *a, **kw: {'status': 'stopping'})
        probe = PeerRunsHTTPClient.probe

        def paused_probe(client, **kwargs):
            result = probe(client, **kwargs)
            probed.set()
            assert publish.wait(10)
            return result

        async def register():
            return await request('groups.peer.register', room_id='linked', member_id='reviewer',
                                 target_url=url, target_profile='default', grant=grant, catalog=catalog)

        monkeypatch.setattr(service, 'stop_room', paused_stop)
        monkeypatch.setattr(PeerRunsHTTPClient, 'probe', paused_probe)
        if phase == 'after_probe':
            seating = asyncio.create_task(register())
            assert await asyncio.to_thread(probed.wait, 10)
        ending = asyncio.create_task(disband(request))
        assert await asyncio.to_thread(reached.wait, 10)
        publish.set()
        if seating is None:
            seating = asyncio.create_task(register())
        result = await asyncio.wait_for(asyncio.shield(seating), timeout=3)
        # A temporary refusal persists a hold, never runnable revocation debt.
        assert _owed(service.db_path) == [("linked", "reviewer", 0)]
        assert hosted_rooms.due_room_grant_revokes(service.db_path, limit=10) == []
        assert (result if surface == 'canonical' else result['message']) == 'room_disbanding_retry'
        assert probed.is_set() == (phase == 'after_probe')
        with pytest.raises(RuntimeStoreError, match='room_disbanding_retry'):
            service._rotate_route_grant(*key, 'not-published', previous=grant)
        assert service.peer_routes == routes and hosted_room_links.load_room_links(service.db_path) == links
        release.set()
        result = await ending
        monkeypatch.setattr(PeerRunsHTTPClient, 'probe', probe)
        if outcome == 'failed':
            assert ('room_stop_pending' if surface == 'canonical' else 'still stopping') in str(result)
            assert _owed(service.db_path) == [("linked", "reviewer", 0)]
            assert await asyncio.to_thread(service.retry_owed_revokes) == 0
            assert hosted_rooms.room_state(service.db_path, room_id='linked').get('disbanded_at') is None
            assert service.peer_routes == routes and hosted_room_links.load_room_links(service.db_path) == links
            route = service.peer_routes[key]
            transport = service._resolve_member_transport(binding, {'payload': {'target_member_id': 'reviewer', 'source_event_seq': 1}})
            dispatch = build_member_dispatch(
                binding=binding, route=route, room_id='linked', task_id='dtask:survivor', target_profile='default',
                execution_generation=1, source_event_seq=1, prompt='review', trace_id=route.trace_id).as_mapping()
            receipt = await asyncio.to_thread(transport.client.dispatch, grant=grant, dispatch=dispatch)
            assert receipt['run_id']
            stored, = hosted_room_links.load_room_links(service.db_path)
            assert stored.grant == service.peer_routes[key].grant != grant
        else:
            assert 'tombstone' in result
            assert service.peer_routes == {} and hosted_room_links.load_room_links(service.db_path) == ()
            # Disband revoked any surviving route at its member gateway first: nothing stays owed.
            assert _owed(service.db_path) == []
            assert await _grant_error(url, grant) == 'room_reauthorization_required'
    finally:
        release.set()
        publish.set()
        await asyncio.gather(*(task for task in (ending, seating) if task is not None), return_exceptions=True)
