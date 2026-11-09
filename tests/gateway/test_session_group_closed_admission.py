"""A closed admission survives every grant alive when Discard confirmed it."""

import asyncio
import time

import pytest

from gateway import hosted_room_driver as driver
from gateway.hosted_room_peer import MAX_STATUS_GRANT_TTL_SECONDS, room_grant_hard_expiry
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_session_group_peer_discard import (
    _never_admitted_deferred_turn, gateway as gateway_fixture, linked_peer as linked_peer_fixture,
)
from tests.gateway.test_session_group_peers import _seat, call
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import build_member_dispatch

gateway, linked_peer = gateway_fixture, linked_peer_fixture


@pytest.mark.asyncio
async def test_closed_key_blocks_delayed_admission_under_older_longer_grant(
        gateway, monkeypatch, linked_peer, tmp_path):
    service, db, identity = await _never_admitted_deferred_turn(gateway, monkeypatch, linked_peer)
    route = service.peer_routes[('linked', 'reviewer')]
    client = service.peer_clients[('linked', 'reviewer')]
    task = driver.get_task(db, identity)
    delayed = build_member_dispatch(
        binding=service.bindings()[0], route=route, room_id='linked', task_id=identity.task_id,
        target_profile='default', execution_generation=1, source_event_seq=task['payload']['source_event_seq'],
        prompt=task['payload']['prompt'], trace_id=route.trace_id).as_mapping()
    _, url, catalog, room = linked_peer
    short_grant, registered = await _seat(gateway, room, url, catalog, ttl_seconds=60)
    assert registered['registered']
    assert (await call(gateway.owner, 'groups.discard', room_id='linked', member_id='reviewer',
                       task_id=identity.task_id, execution_generation=1))['discarded']
    assert driver.get_task(db, identity)['status'] == 'cancelled'
    # Restart the member's store; the original admission has still not arrived.
    gateway.adapter._run_idempotency_store.close()
    gateway.adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    future = room_grant_hard_expiry(short_grant) + 1
    assert future < room_grant_hard_expiry(route.grant)
    monkeypatch.setattr(time, 'time', lambda: future)
    with pytest.raises(PeerRunsHTTPError) as refused:
        await asyncio.to_thread(client.dispatch, dispatch=delayed, grant=route.grant)
    assert refused.value.error_code == 'idempotency_key_conflict'
    assert gateway.adapter._run_statuses == {}


@pytest.mark.parametrize('close_again', [False, True])
def test_closed_keys_expire_after_the_issuers_maximum_status_lifetime(tmp_path, monkeypatch, close_again):
    now = [100.0]
    monkeypatch.setattr(time, 'time', lambda: now[0])
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    try:
        assert store.find_or_close('member', 'attempt') == ('closed', None)
        now[0] += MAX_STATUS_GRANT_TTL_SECONDS - 1
        assert store.lookup('member', 'attempt', 'delayed')[0] == 'conflict'
        if close_again:
            assert store.find_or_close('member', 'attempt') == ('closed', None)
            now[0] += 1
            assert store.lookup('member', 'attempt', 'newer-grant')[0] == 'conflict'
            now[0] += MAX_STATUS_GRANT_TTL_SECONDS - 2
        now[0] += 1
        assert store.lookup('member', 'attempt', 'fresh')[0] == 'missing'
        assert store.reserve('member', 'attempt', 'fresh', 'run', {'status': 'running'})[0] == 'created'
    finally:
        store.close()
