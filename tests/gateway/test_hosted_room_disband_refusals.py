"""Expected Disband refusals keep the group chat and carry stable RPC reasons."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from tests.gateway.test_session_group_peers import (
    _grant_error, _seat, invite, gateway as gateway_fixture, linked_peer as linked_peer_fixture,
)
from tui_gateway import methods_groups, methods_groups_forward, server

gateway = gateway_fixture
linked_peer = linked_peer_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_disband_spaced_identifier_revokes_held_grant(gateway, monkeypatch, linked_peer, surface):
    from tests.gateway.test_hosted_room_disband_publication import _surface
    _, url, catalog, room = linked_peer
    await _seat(gateway, room, url, catalog)
    held = (await invite(gateway, room))['grant']
    gateway.service.hold_grant_revoke(room_id='linked', member_id='reviewer', target_url=url, grant=held)
    request = _surface(gateway, monkeypatch, surface)
    result = await request('groups.disband', room_id=' linked ')
    assert 'tombstone' in result, result
    assert hosted_rooms.held_room_grant_revokes(gateway.db, placement='profile', room_id='linked') == []
    assert await _grant_error(url, held) == 'room_reauthorization_required'
    assert (await request('groups.disband', room_id=' linked '))['tombstone']['idempotent']


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('refusal', ['room_disbanding_retry', 'peer_revoke_unsupported'])
async def test_disband_refusals_are_typed_and_retryable(gateway, monkeypatch, linked_peer, surface, refusal):
    _, url, catalog, room = linked_peer
    assert (await _seat(gateway, room, url, catalog))[1]['registered']
    if surface == 'standalone':
        monkeypatch.setattr(methods_groups, '_service', gateway.service)
        monkeypatch.setattr(methods_groups_forward, '_discover', lambda home: SimpleNamespace(state='absent'))

    async def disband():
        message = {'id': 1, 'method': 'groups.disband', 'params': {'room_id': 'linked'}}
        if surface == 'canonical':
            return await gateway.owner.dispatch(message)
        return await asyncio.to_thread(server.handle_request, message)

    service = gateway.service
    client = service.peer_clients['linked', 'reviewer']
    reached, release = threading.Event(), threading.Event()
    stop = service.stop_room

    def paused_stop(*args, **kwargs):
        reached.set()
        assert release.wait(10)
        return stop(*args, **kwargs)

    first = None
    try:
        if refusal == 'room_disbanding_retry':
            monkeypatch.setattr(service, 'stop_room', paused_stop)
            first = asyncio.create_task(disband())
            assert await asyncio.to_thread(reached.wait, 5)
        else:
            service.peer_clients['linked', 'reviewer'] = SimpleNamespace()
        reply = await disband()
        assert reply['error']['code'] == (4001 if surface == 'canonical' else 4113), reply
        assert reply['error']['data']['reason'] == refusal
        assert reply['error']['message'] == {
            'room_disbanding_retry': 'The group chat is already disbanding; retry deletion.',
            'peer_revoke_unsupported': 'The member cannot revoke its group chat grant; retry after upgrading it.',
        }[refusal]
        assert hosted_rooms.room_state(
            service.store, placement=service.placement, room_id='linked').get('disbanded_at') is None
        assert ('linked', 'reviewer') in service.peer_routes
    finally:
        service.peer_clients['linked', 'reviewer'] = client
        release.set()
        if first is not None:
            await first
    assert (await disband())['result']['tombstone']['disbanded_at'] is not None
