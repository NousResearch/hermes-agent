"""Slow Disband revokes must not block other group chats' route publication."""

import asyncio
import threading

import pytest

from hermes_agent.gateway import hosted_room_links, hosted_rooms
from tests.gateway.test_hosted_room_disband_publication import _surface
from tests.gateway.test_session_group_peers import (
    gateway as gateway_fixture, linked_peer as linked_peer_fixture, linked_room, _seat, invite, call,
)
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient

gateway, linked_peer = gateway_fixture, linked_peer_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_disband_revoke_releases_shared_route_lock(gateway, linked_peer, monkeypatch, surface):
    _, url, catalog, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    grant, registered = await _seat(gateway, room, url, catalog)
    assert registered['registered']
    other = await linked_room(gateway, catalog, room_id='other')
    reached, release = threading.Event(), threading.Event()
    revoke = PeerRunsHTTPClient.revoke_grant

    def paused_revoke(client, **kwargs):
        reached.set()
        assert release.wait(10)
        return revoke(client, **kwargs)

    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', paused_revoke)
    ending = asyncio.create_task(request('groups.disband', room_id='linked'))
    try:
        assert await asyncio.to_thread(reached.wait, 5)
        assert hosted_rooms.room_state(service.db_path, room_id='linked').get('disbanded_at') is None
        assert len(hosted_room_links.load_room_links(service.db_path)) == 1
        refused = await asyncio.wait_for(request(
            'groups.peer.register', room_id='linked', member_id='reviewer', target_url=url,
            target_profile='default', grant=grant, catalog=catalog), 5)
        assert 'room_disbanding_retry' in str(refused)
        _, seated = await asyncio.wait_for(_seat(gateway, other, url, catalog), 5)
        assert seated['registered']
        renewal = (await invite(gateway, other))['grant']
        await asyncio.wait_for(asyncio.to_thread(
            service._rotate_route_grant, 'other', 'reviewer', renewal), 5)
        assert service.peer_routes['other', 'reviewer'].grant == renewal
    finally:
        release.set()
        result = await asyncio.wait_for(ending, 10)
    assert 'tombstone' in result
    assert ('linked', 'reviewer') not in service.peer_routes
    assert (await request('groups.disband', room_id='linked'))['tombstone']['idempotent']
