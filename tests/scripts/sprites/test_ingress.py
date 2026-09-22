import hashlib
import hmac

import pytest
from aiohttp import web, WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from scripts.sprites import ingress

@pytest.fixture(autouse=True)
def isolated_wake_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(ingress, 'WAKE_MARKER', tmp_path / 'wake')


SECRET = 'test-replay-secret-longer-than-32-characters' 
HEADERS = {'X-Hermes-Ingress-Secret': SECRET, 'X-Hermes-Original-Host': 'test.agents.example',
           'X-Hermes-Original-Authorization': 'Bearer user-token'}


@pytest.mark.asyncio
async def test_signed_upload_and_redirect_preserve_application_credentials(monkeypatch, tmp_path):
    seen = []
    body = bytes(range(256)) * 1024
    signature = hmac.new(b'webhook-key', body, hashlib.sha256).hexdigest()

    async def receive(request):
        data = await request.read()
        assert hmac.compare_digest(request.headers['X-Signature'], hmac.new(b'webhook-key', data, hashlib.sha256).hexdigest())
        seen.append((data, request.headers.copy()))
        return web.Response(status=307, headers=[('Location', 'https://test.agents.example/next'),
                            ('Set-Cookie', 'one=1; Secure'), ('Set-Cookie', 'two=2; HttpOnly')])

    upstream = web.Application()
    upstream.router.add_post('/webhooks/test', receive)
    async with TestServer(upstream) as target:
        monkeypatch.setattr(ingress, 'WEBHOOK_PORT', target.port)
        async with TestClient(TestServer(ingress.application(SECRET))) as client:
            denied = await client.post('/webhooks/test', data=body)
            assert denied.status == 401
            result = await client.post('/webhooks/test', data=body, headers={**HEADERS, 'X-Signature': signature,
                                       'Authorization': 'Bearer must-be-removed'}, allow_redirects=False)
            assert result.status == 307
            assert result.headers['Location'] == 'https://test.agents.example/next'
            assert len(result.headers.getall('Set-Cookie')) == 2
    assert len(seen) == 1
    assert seen[0][0] == body
    headers = seen[0][1]
    assert headers['Authorization'] == 'Bearer user-token'
    assert headers['Host'] == 'test.agents.example'
    assert headers['Fly-Replay-Src'] == 'state=' + SECRET
    assert 'X-Hermes-Ingress-Secret' not in headers


@pytest.mark.asyncio
async def test_websocket_duplex_and_streaming_bytes(monkeypatch):
    async def socket(request):
        assert request.headers['Authorization'] == 'Bearer user-token'
        ws = web.WebSocketResponse(protocols=['hermes'])
        await ws.prepare(request)
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                await ws.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await ws.send_bytes(message.data)
        return ws

    async def stream(request):
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'data: first\n\n')
        await response.write(b'data: second\n\n')
        return response

    upstream = web.Application()
    upstream.router.add_get('/api/ws', socket)
    upstream.router.add_get('/stream', stream)
    async with TestServer(upstream) as target:
        monkeypatch.setattr(ingress, 'DASHBOARD_PORT', target.port)
        async with TestClient(TestServer(ingress.application(SECRET))) as client:
            async with client.ws_connect('/api/ws', headers=HEADERS, protocols=['hermes']) as ws:
                assert ws.protocol == 'hermes'
                await ws.send_str('hello')
                assert (await ws.receive()).data == 'hello'
                await ws.send_bytes(b'\x00\xff')
                assert (await ws.receive()).data == b'\x00\xff'
            result = await client.get('/stream', headers=HEADERS)
            assert await result.read() == b'data: first\n\ndata: second\n\n'
