"""Grants received during Disband survive every publication boundary (#149655)."""

import asyncio
from contextlib import nullcontext
import threading
import time

import pytest

from gateway import hosted_room_links, hosted_rooms
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.gateway.test_hosted_room_disband_publication import _surface
from tests.gateway.test_hosted_room_owed_revoke_cap import _fill
from tests.gateway.test_session_group_peers import (
    gateway as gateway_fixture, linked_peer as linked_peer_fixture, linked_room, _seat, invite, _grant_error,
)
from hermes_agent.tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

gateway, linked_peer = gateway_fixture, linked_peer_fixture


def _held(service):
    return {row['grant'] for row in hosted_rooms.held_room_grant_revokes(service.db_path, room_id='linked')}


def _due(service):
    return {row['grant'] for row in hosted_rooms.due_room_grant_revokes(service.db_path, limit=10)}


async def _register(request, url, catalog, grant):
    return await asyncio.wait_for(request(
        'groups.peer.register', room_id='linked', member_id='reviewer', target_url=url,
        target_profile='default', grant=grant, catalog=catalog), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('outcome', ['ended', 'member_refused', 'second_pass_unreachable', 'tombstone_refused', 'restart'])
async def test_grant_arriving_after_sweep_is_durable(gateway, linked_peer, monkeypatch, surface, outcome):
    _, url, catalog, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    route_grant, _ = await _seat(gateway, room, url, catalog)
    reached, release = threading.Event(), threading.Event()
    revoke, tombstone = PeerRunsHTTPClient.revoke_grant, hosted_rooms.disband_room

    def pause_after_revoke(client, *, grant):
        # A newer invitation can be issued after the member acts but before its reply arrives.
        if outcome == 'second_pass_unreachable' and grant != route_grant:
            raise PeerRunsHTTPError('member unavailable', retryable=True)
        if outcome != 'member_refused':
            revoke(client, grant=grant)
        reached.set()
        assert release.wait(10)
        if outcome == 'member_refused':
            raise PeerRunsHTTPError('member unavailable', retryable=True)

    def refuse_tombstone(*args, **kwargs):
        raise hosted_rooms.RoomConflictError('tombstone refused')

    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', pause_after_revoke)
    if outcome in {'tombstone_refused', 'restart'}:
        monkeypatch.setattr(hosted_rooms, 'disband_room', refuse_tombstone)
    ending = asyncio.create_task(request('groups.disband', room_id='linked'))
    try:
        assert await asyncio.to_thread(reached.wait, 5)
        fresh = (await invite(gateway, room))['grant']
        refused = await _register(request, url, catalog, fresh)
        assert 'room_disbanding_retry' in str(refused)
        assert _held(service) == {fresh} and _due(service) == set()
        assert await asyncio.to_thread(service.retry_owed_revokes) == 0
    finally:
        release.set()
        result = await asyncio.wait_for(ending, 10)
    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', revoke)
    monkeypatch.setattr(hosted_rooms, 'disband_room', tombstone)
    if outcome == 'ended':
        assert 'tombstone' in result and _held(service) == set() and _due(service) == set()
        assert await _grant_error(url, fresh) == 'room_reauthorization_required'
    else:
        assert 'tombstone' not in result
        if outcome in {'member_refused', 'second_pass_unreachable'}:
            assert 'peer_unreachable' in str(result)
        assert hosted_rooms.room_state(service.db_path, room_id='linked').get('disbanded_at') is None
        assert _held(service) == {fresh} and _due(service) == set()
        assert await _grant_error(url, fresh) is None
        links = hosted_room_links.load_room_links(service.db_path)
        assert len(links) == (1 if outcome == 'member_refused' else 0)
        if outcome == 'restart':
            # Nothing in the former service can carry the obligation across this restart.
            service = CanonicalHostedRoomService(gateway.authority, None)
            assert service.peer_routes == {} and _held(service) == {fresh}
            service.runtime.status = lambda: {'running': True, 'stopping': False}
            gateway.service = gateway.authority.hosted_room_service = service
            request = _surface(gateway, monkeypatch, surface)
            assert 'tombstone' in await request('groups.disband', room_id='linked')
            assert _held(service) == set() and _due(service) == set()
        else:
            revoked = []

            def record_revoke(client, *, grant):
                revoked.append(grant)
                return revoke(client, grant=grant)

            monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', record_revoke)
            assert 'tombstone' in await request('groups.disband', room_id='linked')
            assert revoked == ([fresh, route_grant] if outcome == 'member_refused' else [fresh])
    assert _held(service) == set() and _due(service) == set()
    assert await _grant_error(url, fresh) == 'room_reauthorization_required'


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_registration_after_tombstone_while_fenced_is_owed(gateway, linked_peer, monkeypatch, surface):
    _, url, catalog, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    assert await asyncio.to_thread(service.retry_owed_revokes) == 0  # park idle maintenance before the hold
    with service.disbanding('linked'):
        hosted_rooms.disband_room(service.db_path, room_id='linked',
            expected_gateway_id=room['authority_gateway_id'], expected_epoch=room['authority_epoch'])
        grant = (await invite(gateway, room))['grant']
        assert 'room_disbanding_retry' in str(await _register(request, url, catalog, grant))
        assert _held(service) == set() and _due(service) == {grant}
    assert (await request('groups.disband', room_id='linked'))['tombstone']['idempotent']
    assert await asyncio.to_thread(service.retry_owed_revokes) == 1
    assert await _grant_error(url, grant) == 'room_reauthorization_required'


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('phase', ['disbanding', 'ended'])
@pytest.mark.parametrize('token', ['other_group', 'unreadable'])
async def test_unprobed_grant_must_name_this_group(gateway, linked_peer, monkeypatch, surface, phase, token):
    _, url, catalog, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    other = await linked_room(gateway, catalog, room_id='other')
    foreign = (await invite(gateway, other))['grant']
    grant = foreign if token == 'other_group' else 'unreadable'
    if phase == 'ended':
        hosted_rooms.disband_room(service.db_path, room_id='linked',
            expected_gateway_id=room['authority_gateway_id'], expected_epoch=room['authority_epoch'])
    with service.disbanding('linked') if phase == 'disbanding' else nullcontext():
        refused = await _register(request, url, catalog, grant)
    assert ('peer_target_mismatch' if surface == 'canonical' else 'group chat grant scope does not match') in str(refused)
    assert _held(service) == set() and _due(service) == set()
    assert await asyncio.to_thread(service.retry_owed_revokes) == 0
    assert await _grant_error(url, foreign) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_successful_registration_removes_its_hold(gateway, linked_peer, monkeypatch, surface):
    _, url, catalog, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    grant = (await invite(gateway, room))['grant']
    with service.disbanding('linked'):
        assert 'room_disbanding_retry' in str(await _register(request, url, catalog, grant))
    assert _held(service) == {grant}
    assert (await _register(request, url, catalog, grant))['registered']
    assert _held(service) == set() and _due(service) == set()
    assert await _grant_error(url, grant) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('limit', ['MAX_OWED_REVOKES', 'MAX_OWED_REVOKES_PER_MEMBER'])
async def test_hold_capacity_failure_is_reported(gateway, linked_peer, monkeypatch, surface, limit):
    _, url, catalog, room = linked_peer
    request = _surface(gateway, monkeypatch, surface)
    first, fresh = [(await invite(gateway, room))['grant'] for _ in range(2)]
    with gateway.service.disbanding('linked'):
        assert 'room_disbanding_retry' in str(await _register(request, url, catalog, first))
        monkeypatch.setattr(hosted_rooms, limit, 1)
        result = await _register(request, url, catalog, fresh)
    assert ('invalid_params' if surface == 'canonical' else 'too many') in str(result), result
    assert _held(gateway.service) == {first} and _due(gateway.service) == set()
    assert await _grant_error(url, first) is None
    assert await _grant_error(url, fresh) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
@pytest.mark.parametrize('reachable', [True, False])
async def test_tombstoned_hold_at_capacity_revokes_outside_locks(
        gateway, linked_peer, monkeypatch, surface, reachable):
    _, url, catalog, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    hosted_rooms.disband_room(service.db_path, room_id='linked',
        expected_gateway_id=room['authority_gateway_id'], expected_epoch=room['authority_epoch'])
    fresh = (await invite(gateway, room))['grant']
    _fill(service.db_path, expires_at=time.time() + 3600)
    reached, release = threading.Event(), threading.Event()
    revoke = PeerRunsHTTPClient.revoke_grant

    def held(client, *, grant):
        reached.set()
        assert release.wait(10)
        if not reachable:
            raise PeerRunsHTTPError('member unavailable', retryable=True)
        return revoke(client, grant=grant)

    monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', held)
    with service.disbanding('linked'):
        registering = asyncio.create_task(_register(request, url, catalog, fresh))
        try:
            assert await asyncio.to_thread(reached.wait, 3)
            def read_under_locks():
                with service.peer_route_lock, service._policy_lock:
                    return _held(service)
            assert await asyncio.wait_for(asyncio.to_thread(read_under_locks), 2) == set()
        finally:
            release.set()
            result = await registering
    expected = 'room_disbanding_retry' if reachable else ('invalid_params' if surface == 'canonical' else 'too many owed revokes')
    assert expected in str(result)
    assert await _grant_error(url, fresh) == ('room_reauthorization_required' if reachable else None)


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_replacement_at_capacity_still_revokes_the_new_grant(gateway, linked_peer, monkeypatch, surface):
    _, url, catalog, room = linked_peer
    request = _surface(gateway, monkeypatch, surface)
    prior, _ = await _seat(gateway, room, url, catalog, ttl_seconds=86400)
    fresh = (await invite(gateway, room))['grant']
    _fill(gateway.service.db_path, expires_at=time.time() + 3600)
    assert 'registered' not in str(await _register(request, url, catalog, fresh))
    for grant in (prior, fresh):
        assert await _grant_error(url, grant) == 'room_reauthorization_required'


@pytest.mark.asyncio
@pytest.mark.parametrize('expired', [False, True])
async def test_tombstone_checks_held_grants_in_its_transaction(gateway, linked_peer, expired):
    import sqlite3

    _, url, _, room = linked_peer
    service = gateway.service
    hosted_rooms.hold_room_grant_revoke(service.db_path, room_id='linked', member_id='reviewer',
        target_url=url, grant='late-grant', expires_at=100, now=50)
    with sqlite3.connect(service.db_path) as conn:
        before = list(conn.iterdump())
    args = dict(room_id='linked', expected_gateway_id=room['authority_gateway_id'],
                expected_epoch=room['authority_epoch'], now=101 if expired else 99)
    if expired:
        assert hosted_rooms.disband_room(service.db_path, **args)['disbanded_at'] == 101
        assert hosted_rooms.disband_room(service.db_path, **args)['idempotent']
        assert not hosted_rooms.owes_room_grant_revokes(service.db_path)
    else:
        with pytest.raises(hosted_rooms.RoomHeldGrantsPendingError) as caught:
            hosted_rooms.disband_room(service.db_path, **args)
        assert caught.value.reason == 'room_disbanding_retry'
        with sqlite3.connect(service.db_path) as conn:
            assert list(conn.iterdump()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['canonical', 'standalone'])
async def test_disband_bounds_repeated_late_grants(gateway, linked_peer, monkeypatch, surface):
    _, url, _, room = linked_peer
    service = gateway.service
    request = _surface(gateway, monkeypatch, surface)
    grants = [(await invite(gateway, room))['grant'] for _ in range(3)]
    tombstone = hosted_rooms.disband_room
    passes = []

    def arriving_before_commit(*args, **kwargs):
        fresh = grants[len(passes)]
        passes.append(fresh)
        service.hold_grant_revoke(room_id='linked', member_id='reviewer', target_url=url, grant=fresh)
        return tombstone(*args, **kwargs)

    monkeypatch.setattr(hosted_rooms, 'disband_room', arriving_before_commit)
    result = await request('groups.disband', room_id='linked')
    assert 'room_disbanding_retry' in str(result)
    assert len(passes) == 3 and _held(service) == {grants[-1]} and _due(service) == set()
    assert hosted_rooms.room_state(service.db_path, room_id='linked').get('disbanded_at') is None
    monkeypatch.setattr(hosted_rooms, 'disband_room', tombstone)
    assert 'tombstone' in await request('groups.disband', room_id='linked')
    assert _held(service) == set()


@pytest.mark.asyncio
async def test_metadata_disband_cannot_release_a_hold_after_its_read(gateway, linked_peer, monkeypatch):
    _, url, _, room = linked_peer
    service = gateway.service
    gateway.authority.hosted_room_service = None
    read = hosted_rooms.room_has_peer_obligations

    def hold_after_read(*args, **kwargs):
        result = read(*args, **kwargs)
        service.hold_grant_revoke(room_id='linked', member_id='reviewer', target_url=url, grant=grant)
        return result

    grant = (await invite(gateway, room))['grant']
    monkeypatch.setattr(hosted_rooms, 'room_has_peer_obligations', hold_after_read)
    request = _surface(gateway, monkeypatch, 'canonical')
    assert await request('groups.disband', room_id='linked') == 'runtime_coordination_required'
    assert hosted_rooms.room_state(service.db_path, room_id='linked').get('disbanded_at') is None
    assert _held(service) == {grant} and _due(service) == set()
