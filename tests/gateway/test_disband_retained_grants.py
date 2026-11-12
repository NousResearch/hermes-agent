"""Disband revokes a retained live grant even after its replacement expires."""

import time

import pytest

from gateway import hosted_room_links, hosted_room_peer, hosted_rooms
from tests.gateway.test_hosted_room_disband_publication import _surface
from tests.gateway.test_session_group_peers import (
    gateway as gateway_fixture, linked_peer as linked_peer_fixture, _seat, _grant_error, _owed)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

gateway, linked_peer = gateway_fixture, linked_peer_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('failure', ['offline', 'lost_ack', 'expired'])
async def test_disband_settles_retained_grants(gateway, linked_peer, monkeypatch, surface, failure):
    _, url, catalog, room = linked_peer
    long_grant, _ = await _seat(gateway, room, url, catalog, ttl_seconds=86400)
    short_grant, _ = await _seat(gateway, room, url, catalog, ttl_seconds=3600)
    now = time.time() + (86401 if failure == 'expired' else 3601)
    monkeypatch.setattr(hosted_room_peer, 'clock', lambda value: now if value is None else value)
    gateway.service.runtime.clock = lambda: now
    assert await _grant_error(url, short_grant) == 'invalid_room_grant'
    assert await _grant_error(url, long_grant) == ('invalid_room_grant' if failure == 'expired' else None)
    request = _surface(gateway, monkeypatch, surface)
    revoke = PeerRunsHTTPClient.revoke_grant
    attempted = []

    def fail_retained(client, *, grant):
        attempted.append(grant)
        if grant == long_grant:
            if failure == 'lost_ack':
                revoke(client, grant=grant)
            raise PeerRunsHTTPError('member unreachable')
        return revoke(client, grant=grant)

    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', fail_retained)
    result = await request('groups.disband', room_id='linked')
    if failure != 'expired':
        assert 'tombstone' not in result
        assert hosted_rooms.room_state(gateway.db.db_path, room_id='linked').get('disbanded_at') is None
        assert len(hosted_room_links.load_room_links(gateway.db.db_path)) == 1
        assert _owed(gateway.db.db_path) == [('linked', 'reviewer', 0)]
        assert await _grant_error(url, long_grant) == (
            'room_reauthorization_required' if failure == 'lost_ack' else None)
        monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', revoke)
        result = await request('groups.disband', room_id='linked')
        assert await _grant_error(url, long_grant) == 'room_reauthorization_required'
    else:
        assert long_grant not in attempted
    assert 'tombstone' in result
    assert hosted_room_links.load_room_links(gateway.db.db_path) == ()
    assert gateway.service.peer_routes == {}
    assert _owed(gateway.db.db_path) == []
