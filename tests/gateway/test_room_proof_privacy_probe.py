"""A substituted endpoint or relay cannot read private canonical peer bodies."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gateway import hosted_room_proof as proof
from gateway.hosted_room_peer import issue_room_grant
from gateway.platforms.api_server_room_proof import wrap
from tests.gateway.test_session_group_peers import gateway  # noqa: F401
from tests.gateway.test_session_group_peer_routes import joined
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


def test_substituted_listener_never_sees_private_dispatch_marker():
    marker = 'PRIVATE_DISPATCH_DO_NOT_DISCLOSE_f18a46bd'
    captured = []
    class Replacement(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            captured.append(self.rfile.read(int(self.headers['Content-Length'])))
            body = b'{"error":{"code":"wrong_installation"}}'
            self.send_response(403)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Replacement)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    grant = issue_room_grant(b'privacy-probe-secret-not-a-credential', grant_id='grant', room_id='room',
        home_install_id='install:home', authority_gateway_id='install:home', authority_epoch=1,
        member_id='reviewer', target_install_id='install:original', target_profile='default',
        execution_policy_digest='a' * 64)
    client = PeerRunsHTTPClient(base_url=f'http://127.0.0.1:{server.server_port}', api_key='',
                               proof_install_id='install:original')
    try:
        with pytest.raises(PeerRunsHTTPError):
            client._request('/v1/runs', method='POST', room_grant=grant,
                            body={'input': marker, 'private_history': [marker]})
        assert captured
        assert marker.encode() not in captured[0], 'replacement received private prompt/history before proof refusal'
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [200, 403])
async def test_relay_cannot_read_request_or_reply_and_proxy_preserves_stop_body(gateway, monkeypatch, status):
    server, _, _, catalog, grant = await joined(gateway, monkeypatch)
    marker = 'PRIVATE_PROMPT_AND_REPLY_84bd3e'
    seen, states = [], []
    @web.middleware
    async def state(request, handler):
        request['original_state'] = 'forwarded'
        response = await handler(request)
        states.append(request.get('handler_state'))
        return response
    async def target_handler(request):
        assert request.body_exists and request.can_read_body
        assert request['original_state'] == 'forwarded'
        request['handler_state'] = 'persisted'
        body = await request.json()
        assert body == {'execution_generation': 7, 'cancel_pending': True, 'input': marker}
        assert request.content_length == len(await request.read())
        assert json.loads(await request.text()) == body
        return web.json_response({'reply': marker}, status=status)
    target_app = web.Application(middlewares=[state])
    target_app.router.add_post('/v1/runs/exact/cancel', wrap(gateway.adapter, target_handler))
    target = TestServer(target_app)
    await target.start_server()
    async def relay(request):
        body = await request.read()
        async with aiohttp.ClientSession() as session:
            async with session.request(request.method, target.make_url(request.raw_path), data=body,
                headers={k: v for k, v in request.headers.items() if k.lower() not in {'host', 'content-length'}}) as reply:
                reply_body = await reply.read()
                seen.append((body, reply_body))
                return web.Response(status=reply.status, body=reply_body,
                    headers={k: v for k, v in reply.headers.items() if k.lower() not in {'content-length', 'transfer-encoding'}})
    relay_app = web.Application()
    relay_app.router.add_route('*', '/{tail:.*}', relay)
    intermediary = TestServer(relay_app)
    await intermediary.start_server()
    client = PeerRunsHTTPClient(base_url=str(intermediary.make_url('')).rstrip('/'), api_key='',
                               proof_install_id=catalog['installation_id'])
    try:
        call = asyncio.to_thread(client._request, '/v1/runs/exact/cancel', method='POST', room_grant=grant,
                                 body={'execution_generation': 7, 'cancel_pending': True, 'input': marker})
        if status == 200:
            assert await call == {'reply': marker}
        else:
            with pytest.raises(PeerRunsHTTPError) as error:
                await call
            assert error.value.status_code == 403
        assert states == ['persisted'] and len(seen) == 1
        assert all(marker.encode() not in wire for wire in seen[0])
    finally:
        await intermediary.close()
        await target.close()
        await server.close()


@pytest.mark.asyncio
async def test_empty_get_and_streaming_refusal_before_handler_effect(gateway, monkeypatch):
    server, _, _, catalog, grant = await joined(gateway, monkeypatch)
    effects = []
    async def target_handler(request):
        effects.append(request.path)
        assert not request.body_exists and not request.can_read_body
        assert request.content_length == 0 and await request.read() == b''
        return web.json_response({'empty': True})
    app = web.Application()
    app.router.add_get('/v1/runs/exact', wrap(gateway.adapter, target_handler))
    app.router.add_get('/v1/runs/exact/events', wrap(gateway.adapter, target_handler))
    target = TestServer(app)
    await target.start_server()
    client = PeerRunsHTTPClient(base_url=str(target.make_url('')).rstrip('/'), api_key='',
                               proof_install_id=catalog['installation_id'])
    try:
        assert await asyncio.to_thread(client._request, '/v1/runs/exact', room_grant=grant) == {'empty': True}
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(client._request, '/v1/runs/exact/events', room_grant=grant)
        assert error.value.status_code == 400 and error.value.error_code == 'room_proof_streaming_unsupported'
        assert effects == ['/v1/runs/exact']
    finally:
        await target.close()
        await server.close()


@pytest.mark.asyncio
async def test_encrypted_wire_overhead_does_not_expand_plaintext_body_limit(gateway, monkeypatch):
    server, _, _, catalog, grant = await joined(gateway, monkeypatch)
    effects = []
    async def handler(request):
        effects.append(await request.read())
        return web.json_response({'size': len(effects[-1])})
    app = web.Application(client_max_size=64)
    app.router.add_post('/v1/runs', wrap(gateway.adapter, handler))
    target = TestServer(app)
    await target.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            for length, status in [(64, 200), (65, 413), (80, 413)]:
                headers = {'Content-Type': 'application/json'}
                auth, key, mac, wire = proof.request_proof(grant, installation_id=catalog['installation_id'],
                    method='POST', path='/v1/runs', body=b'a' * length, headers=headers)
                async with session.post(target.make_url('/v1/runs'), data=wire,
                                        headers={**headers, 'Authorization': auth}) as response:
                    assert response.status == status
                    if status == 200:
                        plain = proof.verify_response(key, mac, response.status, await response.read(),
                            response.headers[proof.RESPONSE_HEADER], response.headers[proof.RESPONSE_NONCE_HEADER])
                        assert json.loads(plain) == {'size': length}
        assert effects == [b'a' * 64]
    finally:
        await target.close()
        await server.close()
