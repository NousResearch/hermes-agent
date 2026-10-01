"""Real HTTP proof transport: replacement endpoints and lost issuance replies."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway import hosted_room_proof as proof
from gateway.platforms.api_server_room_proof import wrap
from tests.gateway.test_session_group_peers import gateway  # noqa: F401
from tests.gateway.test_session_group_peer_routes import joined, capabilities
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.mark.asyncio
async def test_refresh_lost_reply_replays_the_same_grant_but_revocation_blocks_redemption(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    client = PeerRunsHTTPClient(base_url=url, api_key='', proof_install_id=catalog['installation_id'])
    try:
        first = await asyncio.to_thread(client.refresh_grant, grant=grant)
        # Pretend the response vanished before home custody, then construct a new home client.
        restarted = PeerRunsHTTPClient(base_url=url, api_key='', proof_install_id=catalog['installation_id'])
        replay = await asyncio.to_thread(restarted.refresh_grant, grant=grant)
        assert replay['grant'] == first['grant']
        await asyncio.to_thread(client.revoke_grant_exact, grant=grant)
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(restarted.refresh_grant, grant=grant)
        assert error.value.needs_reauthorization
        assert await asyncio.to_thread(capabilities, url, first['grant']) == (200, None)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_same_url_replacement_gets_no_bearer_and_cannot_fabricate_nonadmission(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    port = server.port
    await server.close()
    observed = []
    async def replacement(request):
        observed.append(request.headers['Authorization'])
        return web.json_response({'error': {'code': 'room_reauthorization_required'}}, status=403)
    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', replacement)
    other = TestServer(app, port=port)
    await other.start_server()
    try:
        client = PeerRunsHTTPClient(base_url=url, api_key='', proof_install_id=catalog['installation_id'])
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(client._request, '/v1/runs', method='POST', body={'input': 'frozen'}, room_grant=grant)
        assert error.value.ambiguous and not error.value.not_admitted
        assert observed and all(h.startswith(proof.SCHEME) for h in observed)
        assert grant not in str(observed)
        assert grant.split('.')[1] not in str([json.loads(proof._b64decode(h[len(proof.SCHEME):])) for h in observed])
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_handler_failure_after_effect_is_authenticated_uncertainty(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    effects = []
    async def accepted_then_failed(request):
        effects.append(await request.json())
        raise OSError('response commit failed')
    app = web.Application()
    app.router.add_post('/v1/runs', wrap(gateway.adapter, accepted_then_failed))
    failing = TestServer(app)
    await failing.start_server()
    try:
        client = PeerRunsHTTPClient(base_url=str(failing.make_url('')).rstrip('/'), api_key='',
                                   proof_install_id=catalog['installation_id'])
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(client._request, '/v1/runs', method='POST', body={'input': 'once'}, room_grant=grant)
        assert effects == [{'input': 'once'}]
        assert error.value.status_code == 503 and error.value.ambiguous and not error.value.not_admitted
    finally:
        await failing.close()
        await server.close()


@pytest.mark.asyncio
async def test_expired_dispatch_can_clean_lost_issuance_without_redeeming_it(gateway, monkeypatch):
    import time
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret, issue_room_grant
    from gateway import hosted_rooms
    server, url, room, catalog, original = await joined(gateway, monkeypatch)
    claims = decode_room_grant(gateway_room_grant_secret(), original, permission='status')
    scope = {k: claims[k] for k in ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
                                    'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest')}
    old = issue_room_grant(gateway_room_grant_secret(), grant_id='short-issuance', **scope,
                          issued_at=time.time(), ttl_seconds=.3, status_ttl_seconds=3600)
    client = PeerRunsHTTPClient(base_url=url, api_key='', proof_install_id=catalog['installation_id'])
    body = b'{"ttl_seconds":3600}'
    request_id = proof.issuance_request_id(old, body)
    try:
        issued = await asyncio.to_thread(client._scoped_post, '/v1/room-members/grants/refresh', old,
                                         body={'ttl_seconds': 3600})
        await asyncio.sleep(.35)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client._scoped_post, '/v1/room-members/grants/refresh', old,
                                    body={'ttl_seconds': 3600})
        result = await asyncio.to_thread(client.cleanup_issuance, grant=old, request_id=request_id)
        assert result == {'revoked': True} and 'grant' not in result
        assert await asyncio.to_thread(capabilities, url, issued['grant']) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(client.cleanup_issuance, grant=old, request_id=request_id) == result
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_cleanup_tombstones_delayed_issuance_and_revoked_old_cannot_kill_successor(gateway, monkeypatch):
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    client = PeerRunsHTTPClient(base_url=url, api_key='', proof_install_id=catalog['installation_id'])
    try:
        request_id = proof.issuance_request_id(grant, b'{"ttl_seconds":1234}')
        await asyncio.to_thread(client.cleanup_issuance, grant=grant, request_id=request_id)
        # A lost cleanup acknowledgement is idempotent even when minting never happened.
        await asyncio.to_thread(client.cleanup_issuance, grant=grant, request_id=request_id)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client._scoped_post, '/v1/room-members/grants/refresh', grant,
                                    body={'ttl_seconds': 1234})
        issued = await asyncio.to_thread(client.refresh_grant, grant=grant, ttl_seconds=3600)
        await asyncio.to_thread(client.revoke_grant_exact, grant=grant)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client.cleanup_issuance, grant=grant,
                                    request_id=proof.issuance_request_id(grant, b'{"ttl_seconds":3600}'))
        assert await asyncio.to_thread(capabilities, url, issued['grant']) == (200, None)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_pending_issuance_receipt_can_be_cleaned_after_handler_crash(gateway, monkeypatch):
    from gateway.platforms import api_server_room_grants
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    client = PeerRunsHTTPClient(base_url=url, api_key='', proof_install_id=catalog['installation_id'])
    original = api_server_room_grants._handle_room_member_grant_refresh
    async def crash(*args, **kwargs):
        raise OSError('crash after receipt preparation before response commit')
    try:
        monkeypatch.setattr(api_server_room_grants, '_handle_room_member_grant_refresh', crash)
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(client.refresh_grant, grant=grant, ttl_seconds=3600)
        assert error.value.ambiguous
        monkeypatch.setattr(api_server_room_grants, '_handle_room_member_grant_refresh', original)
        request_id = proof.issuance_request_id(grant, b'{"ttl_seconds":3600}')
        assert await asyncio.to_thread(client.cleanup_issuance, grant=grant, request_id=request_id) == {'revoked': True}
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client.refresh_grant, grant=grant, ttl_seconds=3600)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_home_journals_issuance_before_lost_refresh_reply_and_cleans_after_restart(gateway, monkeypatch):
    from gateway import hosted_room_peer, session_group_peer_cleanup as cleanup
    from gateway.session_group_peer_routes import CanonicalPeerClient
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    key = ('linked', 'reviewer')
    raw, route = gateway.service.peer_clients[key], gateway.service.peer_routes[key]
    client = CanonicalPeerClient(gateway.service, HostedRoomBinding('linked', room['authority_gateway_id'], 1),
                                 key, route, raw)
    issued = []
    original = raw._scoped_post
    def lose_reply(path, bearer, **kwargs):
        result = original(path, bearer, **kwargs)
        if path.endswith('/refresh'):
            issued.append(result['grant'])
            raise PeerRunsHTTPError('lost issuance response', ambiguous=True)
        return result
    try:
        monkeypatch.setattr(raw, '_scoped_post', lose_reply)
        monkeypatch.setattr(hosted_room_peer, 'room_grant_needs_dispatch_refresh', lambda *args, **kwargs: True)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client._refresh_if_due, 'probe', grant, {})
        assert issued and cleanup.status(gateway.service.db_path)[0]['mode'] == 'issuance'
        restarted = CanonicalHostedRoomService(gateway.authority, None)
        await asyncio.to_thread(cleanup.drain, restarted, force=True)
        assert cleanup.status(gateway.service.db_path) == []
        assert await asyncio.to_thread(capabilities, url, issued[0]) == (403, 'room_reauthorization_required')
        assert await asyncio.to_thread(capabilities, url, grant) == (200, None)
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('duplicate', ['Idempotency-Key', 'iDeMpOtEnCy-KeY', 'Authorization'])
async def test_duplicate_effect_headers_cannot_change_a_signed_request(gateway, monkeypatch, duplicate):
    import aiohttp
    server, url, room, catalog, grant = await joined(gateway, monkeypatch)
    effects = []
    async def accept(request):
        effects.append(request.headers.get('Idempotency-Key'))
        return web.json_response({'run_id': 'accepted'}, status=202)
    app = web.Application()
    app.router.add_post('/v1/runs', wrap(gateway.adapter, accept))
    target = TestServer(app)
    await target.start_server()
    body = b'{"input":"once"}'
    bound = {'Content-Type': 'application/json', 'Idempotency-Key': 'correct'}
    authorization, _, _ = proof.request_proof(grant, installation_id=catalog['installation_id'],
        method='POST', path='/v1/runs', body=body, headers=bound)
    headers = [('Authorization', authorization), *bound.items()]
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(target.make_url('/v1/runs'), data=body, headers=headers) as response:
                assert response.status == 202
            # The prior bug authenticated the LAST duplicate while the handler read the FIRST.
            forged = [(duplicate, authorization if duplicate == 'Authorization' else 'other'), *headers]
            # Raw HTTP preserves case-varied duplicates that ClientSession may coalesce.
            reader, writer = await asyncio.open_connection('127.0.0.1', target.port)
            request = ('POST /v1/runs HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n'
                       + ''.join(f'{key}: {value}\r\n' for key, value in forged)
                       + f'Content-Length: {len(body)}\r\n\r\n').encode() + body
            writer.write(request)
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            assert int(response.split(b' ', 2)[1]) == 401, response
        assert effects == ['correct']
    finally:
        await target.close()
        await server.close()
