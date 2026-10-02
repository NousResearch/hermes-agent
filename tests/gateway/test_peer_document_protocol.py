"""A text-only peer rejects the document extension without receiving file bytes."""
import asyncio
import copy

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
from gateway.platforms.api_server_room_proof import wrap
from tests.gateway.test_session_group_peers import gateway  # noqa: F401
from tests.gateway.test_session_group_peer_routes import joined
from tests.tui_gateway.test_hosted_room_peer_http import _dispatch
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


@pytest.mark.asyncio
async def test_older_peer_keeps_text_wire_and_never_receives_document_bytes(gateway, monkeypatch):
    server, _, _, catalog, grant = await joined(gateway, monkeypatch)
    claims = decode_room_grant(gateway_room_grant_secret(), grant, permission='dispatch')
    fields = ('room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch', 'member_id', 'target_install_id', 'target_profile')
    text = _dispatch(**{k: claims[k] for k in fields}, capability_digest=catalog['catalog_digest'],
                     execution_policy_digest=catalog['execution_policy']['policy_digest'])
    bodies = []
    async def older_runs(request):
        body = await request.json()
        bodies.append(body)
        assert 'document_bytes' not in body
        if 'document_inputs' in body['hosted_room_dispatch']:
            return web.json_response({'error': {'code': 'invalid_room_dispatch'}}, status=400)
        assert body == {'input': text['prompt'], 'hosted_room_dispatch': text}
        return web.json_response({'run_id': 'text-run', 'status': 'queued'}, status=202)
    app = web.Application()
    app.router.add_post('/v1/runs', wrap(gateway.adapter, older_runs))
    target = TestServer(app)
    await target.start_server()
    try:
        client = PeerRunsHTTPClient(base_url=str(target.make_url('')).rstrip('/'), api_key='', proof_install_id=catalog['installation_id'])
        assert (await asyncio.to_thread(client.dispatch, dispatch=text, grant=grant))['status'] == 'accepted'
        documents = {**text, 'task_id': 'document-attempt', 'document_inputs': [{
            'event_id': 'source-event', 'attachment_id': 'att_' + '1'*32, 'recipient_member_id': claims['member_id'],
            'kind': 'file', 'name': 'a.txt', 'mime': 'text/plain', 'size': 1, 'sha256': 'a'*64}]}
        with pytest.raises(PeerRunsHTTPError) as fresh:
            await asyncio.to_thread(client.dispatch, dispatch=documents, grant=grant)
        assert fresh.value.not_admitted and fresh.value.error_code == 'invalid_room_dispatch'
        with pytest.raises(PeerRunsHTTPError) as recovery:
            await asyncio.to_thread(client.recover_dispatch, dispatch=documents, grant=grant)
        assert recovery.value.ambiguous and not recovery.value.not_admitted
        count = len(bodies)
        with pytest.raises(PeerRunsHTTPError) as delayed:
            await asyncio.to_thread(client.recover_dispatch, dispatch=documents, grant=grant)
        assert delayed.value.ambiguous and delayed.value.retryable
        assert len(bodies) == count
        oversized = copy.deepcopy(documents)
        oversized['document_inputs'][0]['size'] = 5_000_001
        with pytest.raises(ValueError):
            await asyncio.to_thread(client.dispatch, dispatch=oversized, grant=grant)
        assert len(bodies) == count
    finally:
        await target.close()
        await server.close()

@pytest.mark.asyncio
async def test_document_proof_budget_does_not_raise_or_lower_ordinary_api_limit(gateway, monkeypatch):
    from aiohttp import ClientSession
    from gateway import hosted_room_proof as proof
    from gateway.hosted_room_documents import DOCUMENT_HTTP_MAX_BYTES
    from gateway.platforms.api_server import MAX_REQUEST_BYTES

    server, _, _, catalog, grant = await joined(gateway, monkeypatch)
    async def consume(request):
        return web.json_response({'bytes': len(await request.read())})
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app.router.add_post('/v1/runs', wrap(gateway.adapter, consume, max_bytes=DOCUMENT_HTTP_MAX_BYTES))
    target = TestServer(app)
    await target.start_server()
    try:
        async with ClientSession() as client:
            # This body fits the unchanged ordinary API ceiling but not the smaller proof budget.
            body = b'x' * (DOCUMENT_HTTP_MAX_BYTES + 1)
            async with client.post(target.make_url('/v1/runs'), data=body) as response:
                assert response.status == 200
                assert (await response.json())['bytes'] == len(body)
            async with client.post(target.make_url('/v1/runs'), data=b'x' * (MAX_REQUEST_BYTES + 1)) as response:
                assert response.status == 413
            authorization, _, _, wire = proof.request_proof(grant,
                installation_id=catalog['installation_id'], method='POST', path='/v1/runs', body=body)
            async with client.post(target.make_url('/v1/runs'), data=wire,
                                   headers={'Authorization': authorization}) as response:
                assert response.status == 413
    finally:
        await target.close()
        await server.close()
