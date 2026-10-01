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
