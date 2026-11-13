"""Discard of a deferred peer turn whose home lost its receipt never starts that turn on the
member's gateway: the home asks which run the turn's admission key started (closing the key when
it started none) instead of replaying the admission, which would start the turn there."""
import asyncio
import sqlite3
import time

import pytest
from aiohttp import web

from gateway import hosted_room_driver as driver
from gateway.platforms import api_server_room_grants
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.gateway import test_session_group_peers as peers

call, invite = peers.call, peers.invite
gateway, linked_peer = peers.gateway, peers.linked_peer  # the shared two-role gateway fixtures


async def _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer):
    """The home's POST of the turn was lost before the member's gateway admitted it (the member
    has no run for it), the home restarted, and the turn was deferred."""
    from gateway.platforms import api_server_runs

    async def inert(adapter, run, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    _server, url, catalog, room = linked_peer
    grant = (await invite(gateway, room))['grant']
    assert (await call(gateway.owner, 'groups.peer.register', room_id='linked', member_id='reviewer',
                       target_url=url, target_profile='default', grant=grant, catalog=catalog))['registered']
    assert (await call(gateway.owner, 'groups.send', room_id='linked', event_id='ask',
                       payload={'text': '@reviewer review this', 'thread_id': 'thread'}))['accepted']
    db, service = gateway.db.db_path, gateway.service
    task, = driver.list_tasks(db, room_id='linked', status='queued')
    identity, binding = task['identity'], service.bindings()[0]

    def earlier():
        return time.time() - 30
    crashed = driver.acquire_lease(db, room_id='linked', gateway_id=binding.gateway_id,
                                   authority_epoch=binding.authority_epoch, process_generation='crashed',
                                   ttl_seconds=1, clock=earlier)
    driver.start_task(db, identity, crashed, expected_cancel_generation=0, clock=earlier)
    service = gateway.authority.hosted_room_service = CanonicalHostedRoomService(gateway.authority, None)
    idle = service.runtime.status
    service.runtime.status = lambda: {**idle(), 'running': True, 'stopping': False}
    lease = service.runtime._ensure_lease(binding)
    driver.recover_room(db, lease, clock=time.time)
    service.runtime._fenced(driver.defer_indeterminate_task, None, driver.get_task(db, identity), lease,
                            reason='member_unavailable')
    assert gateway.adapter._run_statuses == {}
    return service, db, identity


def _member_runs(tmp_path):
    with sqlite3.connect(tmp_path / 'runs.db') as runs:
        return runs.execute('SELECT idempotency_key, status_json FROM run_idempotency').fetchall()


@pytest.mark.asyncio
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_discard_of_a_turn_the_member_never_admitted_settles_it_without_starting_it(
        gateway, monkeypatch, linked_peer, tmp_path):
    service, db, identity = await _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer)
    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    discarded = await call(gateway.owner, 'groups.discard', **control)
    assert isinstance(discarded, dict) and discarded['discarded'], discarded
    assert driver.get_task(db, identity)['status'] == 'cancelled'
    assert gateway.adapter._run_statuses == {}, 'Discard started the turn on the member gateway'
    # The key is closed on the member: the home's lost POST, arriving late, is refused there.
    from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
    from hermes_agent.tui_gateway.hosted_room_peer_transport import build_member_dispatch
    route, client = service.peer_routes[('linked', 'reviewer')], service.peer_clients[('linked', 'reviewer')]
    task = driver.get_task(db, identity)
    late = build_member_dispatch(
        binding=service.bindings()[0], route=route, room_id='linked', task_id=identity.task_id,
        target_profile='default', execution_generation=1, source_event_seq=task['payload']['source_event_seq'],
        prompt=task['payload']['prompt'], trace_id=route.trace_id)
    with pytest.raises(PeerRunsHTTPError) as refused:
        await asyncio.to_thread(client.dispatch, grant=route.grant, dispatch=late.as_mapping())
    assert refused.value.error_code == 'idempotency_key_conflict'
    assert gateway.adapter._run_statuses == {}
    assert [key for key, _ in _member_runs(tmp_path)] == [f'room:{identity.task_id}:1']
    assert 'tombstone' in await call(gateway.owner, 'groups.disband', room_id='linked')


@pytest.mark.asyncio
@pytest.mark.parametrize('linked_peer', [True], indirect=True)
async def test_discard_of_an_admitted_turn_finds_its_run_without_replaying_the_admission(
        gateway, monkeypatch, linked_peer):
    """The member admitted the turn and the home lost its receipt: Discard finds that run by its key,
    stops it, and settles only once the member confirms; no admission POST is sent to find it."""
    _service, db, identity = await peers._deferred_unreceipted_peer_turn(gateway, monkeypatch, linked_peer)
    adapter = gateway.adapter
    run_id, = list(adapter._run_statuses)
    replays = []
    lookup = adapter._run_idempotency_store.lookup
    monkeypatch.setattr(adapter._run_idempotency_store, 'lookup',
                        lambda *a, **kw: replays.append(a[1]) or lookup(*a, **kw))
    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    assert await call(gateway.owner, 'groups.discard', **control) == 'session_busy'
    assert adapter._run_statuses[run_id]['status'] == 'stopping'
    assert driver.get_task(db, identity)['status'] == 'deferred'
    adapter._set_run_status(run_id, 'cancelled')  # the member's run ends
    discarded = await call(gateway.owner, 'groups.discard', **control)
    assert isinstance(discarded, dict) and discarded['discarded'], discarded
    assert driver.get_task(db, identity)['status'] == 'cancelled'
    assert list(adapter._run_statuses) == [run_id]
    assert replays == [], 'Discard replayed the admission POST to find the run'


@pytest.mark.asyncio
async def test_discard_keeps_the_turn_on_a_member_that_cannot_answer_without_starting_it(
        gateway, monkeypatch, linked_peer):
    """A member gateway too old to look a key up (no such route) or unreachable keeps the turn,
    and the home never falls back to the admitting POST."""
    _service, db, identity = await _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer)

    async def no_route(adapter, request):
        return web.Response(status=404, text='404: Not Found')
    monkeypatch.setattr(api_server_room_grants, '_handle_room_member_run_close', no_route, raising=False)
    control = dict(room_id='linked', member_id='reviewer', task_id=identity.task_id, execution_generation=1)
    reply = await gateway.owner.dispatch({'id': 1, 'method': 'groups.discard', 'params': control})
    assert reply['error']['message'] == 'peer_unreachable', reply
    assert 'too old' in str(reply['error'].get('data')), reply
    assert driver.get_task(db, identity)['status'] == 'deferred'
    assert gateway.adapter._run_statuses == {}, 'Discard started the turn on the member gateway'
    await linked_peer[0].close()
    assert await call(gateway.owner, 'groups.discard', **control) == 'peer_unreachable'
    assert driver.get_task(db, identity)['status'] == 'deferred'
    assert gateway.adapter._run_statuses == {}
