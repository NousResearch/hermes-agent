"""Discard and Disband resolve a restarted member's exact lost admission over HTTP."""

import asyncio

import pytest

from hermes_agent.gateway import hosted_room_driver as driver, hosted_rooms
from hermes_agent.gateway.run import _profile_runtime_scope
from hermes_agent.hermes_state_runtime import list_session_admissions
from tests.gateway.test_api_unknown_resolution import (
    _AUTH_HEADERS, _close_fixture, _restarted_api_runs,
)
from tests.gateway import test_session_group_peers as peers
from tests.waiting import async_wait_until
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from hermes_agent.tui_gateway.hosted_room_peer_transport import build_member_dispatch

gateway, linked_peer = peers.gateway, peers.linked_peer


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['discard', 'disband'])
@pytest.mark.parametrize('fault', ['none', 'resolve', 'run_id', 'admission_id', 'generation', 'offline'])
async def test_restart_stop_requires_the_members_exact_confirmation(
        gateway, linked_peer, monkeypatch, tmp_path, operation, fault):
    """A real member owner restart leaves an unknown head and a queued follower.

    Seed the home's deferred attempt with that admitted run's receipt; the existing
    peer-HTTP fixture supplies the route and grant revocation for canonical controls.
    """
    _, url, catalog, room = linked_peer
    assert (await peers._seat(gateway, room, url, catalog))[1]['registered']
    service, db = gateway.service, gateway.db.db_path
    assert (await peers.call(gateway.owner, 'groups.send', room_id='linked', event_id='ask',
                            payload={'text': '@reviewer review', 'thread_id': 'thread'}))['accepted']
    task, = driver.list_tasks(db, room_id='linked', status='queued')
    identity, binding = task['identity'], service.bindings()[0]
    now = [100.0]
    service.runtime.clock = lambda: now[0]
    lease = driver.acquire_lease(db, room_id='linked', gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation='earlier-home',
        ttl_seconds=1, clock=service.runtime.clock)
    driver.start_task(db, identity, lease, expected_cancel_generation=0, clock=service.runtime.clock)
    now[0] = 102.0
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(db, lease, clock=service.runtime.clock)
    driver.defer_indeterminate_task(db, identity, lease, expected_execution_generation=1,
        expected_cancel_generation=0, reason='member_unavailable', clock=service.runtime.clock)
    peer, route = service.peer_clients[('linked', 'reviewer')], service.peer_routes[('linked', 'reviewer')]
    dispatch = build_member_dispatch(binding=binding, route=route, room_id='linked',
        task_id=identity.task_id, target_profile='default', execution_generation=1,
        source_event_seq=task['payload']['source_event_seq'], prompt=task['payload']['prompt'],
        trace_id=route.trace_id)
    peer.bind_room_scope(**{key: getattr(dispatch, key) for key in (
        'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
        'member_id', 'target_install_id', 'target_profile')})
    member_home = tmp_path / 'member'
    member_home.mkdir()
    client, adapters, store, owner, unknown, follower, executed, tasks = (
        await _restarted_api_runs(member_home, monkeypatch))
    monkeypatch.setenv('HERMES_HOME', str(gateway.home))
    peer._save_receipt(dispatch, run_id=unknown['request_id'], session_id='api-session')
    run_path = '/v1/runs/' + unknown['request_id']
    original, requests = peer._request, []
    loop = asyncio.get_running_loop()
    broken = [fault]

    def forward(path, *, method='GET', body=None, **kwargs):
        requests.append((method, path, body))
        if not path.startswith('/v1/runs/'):
            return original(path, method=method, body=body, **kwargs)
        if broken[0] == 'offline' or (broken[0] == 'resolve' and path.endswith('/resolve-unknown')):
            raise PeerRunsHTTPError('member response lost', retryable=True)

        async def request():
            with _profile_runtime_scope(member_home):
                response = await client.request(method, path, json=body, headers=_AUTH_HEADERS)
                result = await response.json()
                if response.status >= 400:
                    raise PeerRunsHTTPError(str(result), status_code=response.status,
                                            error_code=result['error']['code'])
                if method == 'GET' and result['status'] == 'unknown':
                    if broken[0] in {'run_id', 'admission_id'}:
                        result[broken[0]] = 'different-admission'
                    elif broken[0] == 'generation':
                        result['execution_generation'] += 1
                return result
        return asyncio.run_coroutine_threadsafe(request(), loop).result(10)

    monkeypatch.setattr(peer, '_request', forward)
    # An observation error is still in backoff when the member returns. Stop must
    # fetch fresh evidence instead of repeating that cached error or running status.
    peer.clock = lambda: 0.0
    record = peer._receipt(identity.task_id, 1)
    saved_fault, broken[0] = broken[0], 'offline'
    try:
        with pytest.raises(PeerRunsHTTPError, match='response lost'):
            await asyncio.to_thread(peer._poll_receipt, record, grant=route.grant)
        broken[0] = saved_fault
        params = {'room_id': 'linked'}
        if operation == 'discard':
            params.update(member_id='reviewer', task_id=identity.task_id, execution_generation=1)
        try:
            result = await peers.call(gateway.owner, 'groups.' + operation, **params)
        except RuntimeError as exc:
            # The base predates the separate typed Disband-refusal fix.
            assert fault != 'none' and 'still stopping' in str(exc)
            result = str(exc)
        if fault != 'none':
            assert not isinstance(result, dict), result
            assert driver.get_task(db, identity)['status'] == 'deferred'
            assert hosted_rooms.room_state(db, room_id='linked').get('disbanded_at') is None
            rows = list_session_admissions(owner.db, session_id='api-session')
            assert [row['status'] for row in rows] == ['unknown', 'queued']
            assert executed == []
            if fault == 'run_id':
                assert not any(path.endswith('/resolve-unknown') for _, path, _ in requests)
            broken[0] = 'none'
            result = await peers.call(gateway.owner, 'groups.' + operation, **params)
        assert isinstance(result, dict), result
        assert result.get('discarded') if operation == 'discard' else 'tombstone' in result
        settled = driver.get_task(db, identity)
        assert (settled['status'], settled['execution_generation']) == ('cancelled', 1)
        assert (identity.task_id, 1) in peer._terminal_receipts
        await async_wait_until(lambda: all(row['status'] == 'terminal' for row in
            list_session_admissions(owner.db, session_id='api-session', pending_only=False)),
            what='member follower completes after unknown acknowledgement')
        assert executed == ['FOLLOWER']
        rows = list_session_admissions(owner.db, session_id='api-session', pending_only=False)
        assert {row['admission_id'] for row in rows} == {unknown['admission_id'], follower['admission_id']}
        assert not any(path == '/v1/runs' for _, path, _ in requests)
        assert gateway.adapter._run_statuses == {}
        exact = {'admission_id': unknown['admission_id'], 'execution_generation': unknown['generation']}
        assert ('POST', run_path + '/resolve-unknown', exact) in requests
        # The same acknowledgement replays safely after the follower has completed.
        await asyncio.to_thread(forward, run_path + '/resolve-unknown', method='POST', body=exact)
        assert executed == ['FOLLOWER']
        if operation == 'discard':
            assert (await peers.call(gateway.owner, 'groups.discard', **params))['discarded']
    finally:
        with _profile_runtime_scope(member_home):
            await _close_fixture(client, adapters, store, tasks)
