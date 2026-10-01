"""Revocation survives route retirement, restart, offline targets and lost replies."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_links as links
from gateway import session_group_peer_cleanup as cleanup
from gateway.session_hosted_service import CanonicalHostedRoomService
from tests.gateway.test_session_group_peers import gateway  # noqa: F401
from tests.gateway.test_session_group_peer_routes import joined, capabilities, reregister
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.mark.asyncio
async def test_disband_cleanup_survives_offline_target_and_home_restart(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    original = PeerRunsHTTPClient.revoke_grant
    try:
        def offline(*args, **kwargs):
            raise PeerRunsHTTPError('offline', retryable=True)
        monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', offline)
        await asyncio.to_thread(gateway.service.revoke_room_routes, 'linked')
        assert not links.load_room_links(gateway.service.db_path)
        assert cleanup.status(gateway.service.db_path, 'linked') == [
            {'room_id': 'linked', 'member_id': 'reviewer', 'mode': 'scope', 'status': 'pending', 'attempts': 1}]
        assert await asyncio.to_thread(capabilities, url, grant) == (200, None)
        restarted = CanonicalHostedRoomService(gateway.authority, None)
        monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant', original)
        await asyncio.to_thread(cleanup.drain, restarted, force=True)
        assert cleanup.status(restarted.db_path) == []
        assert await asyncio.to_thread(capabilities, url, grant) == (403, 'room_reauthorization_required')
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_lost_exact_revoke_reply_keeps_obligation_and_successor(gateway, monkeypatch):
    server, url, room, catalog, old = await joined(gateway, monkeypatch)
    original = PeerRunsHTTPClient.revoke_grant_exact
    try:
        old_link = links.load_room_link(gateway.service.db_path, room_id='linked', member_id='reviewer')
        new, _ = await reregister(gateway, room, url, catalog)
        cleanup.retain(gateway.service.db_path, old_link)
        def lose_reply(client, **kwargs):
            original(client, **kwargs)
            raise PeerRunsHTTPError('reply disappeared', ambiguous=True)
        monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant_exact', lose_reply)
        await asyncio.to_thread(cleanup.drain, gateway.service, force=True)
        assert cleanup.status(gateway.service.db_path)
        assert await asyncio.to_thread(capabilities, url, new) == (200, None)
        restarted = CanonicalHostedRoomService(gateway.authority, None)
        monkeypatch.setattr(PeerRunsHTTPClient, 'revoke_grant_exact', original)
        await asyncio.to_thread(cleanup.drain, restarted, force=True)
        assert cleanup.status(restarted.db_path) == []
        assert await asyncio.to_thread(capabilities, url, old) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(capabilities, url, new) == (200, None)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_failed_renewal_publication_retains_returned_grant_cleanup(gateway, monkeypatch):
    from tests.gateway.test_session_group_peer_routes import KEY
    from gateway.session_group_peer_routes import CanonicalPeerClient
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    server, url, room, catalog, old = await joined(gateway, monkeypatch)
    try:
        route = gateway.service.peer_routes[KEY]
        raw = gateway.service.peer_clients[KEY]
        new = (await asyncio.to_thread(raw.refresh_grant, grant=old))['grant']
        binding = HostedRoomBinding('linked', room['authority_gateway_id'], room['authority_epoch'])
        client = CanonicalPeerClient(gateway.service, binding, KEY, route, raw)
        def fail_write(*args, **kwargs):
            raise OSError('disk write failed')
        monkeypatch.setattr(gateway.service, '_save_link', fail_write)
        monkeypatch.setattr(raw, 'revoke_grant_exact', fail_write)
        with pytest.raises(OSError, match='disk write failed'):
            await asyncio.to_thread(client._publish_renewal, old, {'grant': new})
        assert links.load_room_link(gateway.service.db_path, room_id='linked', member_id='reviewer').grant == old
        assert cleanup.status(gateway.service.db_path)[0]['status'] == 'pending'
        restarted = CanonicalHostedRoomService(gateway.authority, None)
        await asyncio.to_thread(cleanup.drain, restarted, force=True)
        assert await asyncio.to_thread(capabilities, url, new) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(capabilities, url, old) == (200, None)
    finally:
        await server.close()


def test_corrupt_cleanup_is_visible_and_never_deleted(gateway):
    gateway.db._execute_write(lambda conn: conn.execute(
        'INSERT INTO hosted_room_peer_cleanup(key,value) VALUES (?,?)', (cleanup._PREFIX + 'broken', 'not json')))
    cleanup.drain(gateway.service)
    assert cleanup.status(gateway.service.db_path) == [{'status': 'unreadable'}]


@pytest.mark.asyncio
async def test_due_cleanup_after_corrupt_and_backoff_prefix_is_not_starved(gateway, monkeypatch):
    import json
    import time
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    try:
        link = links.load_room_link(gateway.service.db_path, room_id='linked', member_id='reviewer')
        with gateway.db._read_ctx() as conn:
            assert conn.execute('SELECT COUNT(*) FROM hosted_room_peer_cleanup').fetchone()[0] == 0
        def seed(conn):
            for i in range(40):
                value = 'unreadable' if i % 2 else json.dumps(
                    {'link': link.as_record(), 'mode': 'exact', 'attempts': 1, 'next_at': time.time() + 300})
                conn.execute('INSERT INTO hosted_room_peer_cleanup(key,value) VALUES (?,?)',
                             (cleanup._PREFIX + f'000{i:03}', value))
        gateway.db._execute_write(seed)
        key = cleanup.retain(gateway.service.db_path, link)
        await asyncio.to_thread(cleanup.drain, gateway.service)
        assert key not in {key for key, _ in cleanup.obligations(gateway.service.db_path)}
        assert len(cleanup.obligations(gateway.service.db_path)) == 40
        assert await asyncio.to_thread(capabilities, url, grant) == (403, 'room_reauthorization_required')
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_pending_disband_fences_send_and_route_publication_and_resumes_after_restart(gateway, monkeypatch):
    from gateway import hosted_rooms
    from hermes_state_runtime import RuntimeStoreError
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    try:
        gateway.service.begin_disband('linked')
        def still_stopping(*args, **kwargs):
            raise RuntimeError('target still running')
        monkeypatch.setattr(gateway.service, 'stop_room', still_stopping)
        await asyncio.to_thread(gateway.service._resume_disbands)
        assert gateway.service.is_retiring('linked')
        with pytest.raises(RuntimeStoreError, match='room_retiring'):
            gateway.service.send(room_id='linked', event_id='too-late', payload={'text': 'must not execute'})
        with pytest.raises(RuntimeStoreError, match='room_retiring'):
            gateway.service.register_peer_route(room_id='linked', member_id='reviewer',
                route=gateway.service.peer_routes[('linked', 'reviewer')],
                client=gateway.service.peer_clients[('linked', 'reviewer')], target_url=url,
                catalog=links.load_room_link(gateway.service.db_path, room_id='linked', member_id='reviewer').catalog)
        restarted = CanonicalHostedRoomService(gateway.authority, None)
        assert restarted.is_retiring('linked')
        await asyncio.to_thread(restarted._resume_disbands)
        assert hosted_rooms.room_state(restarted.db_path, room_id='linked', include_disbanded=True)['disbanded_at']
        assert await asyncio.to_thread(capabilities, url, grant) == (403, 'room_reauthorization_required')
    finally:
        await server.close()
