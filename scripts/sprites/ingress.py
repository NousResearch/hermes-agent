"""Multiplex the native Sprites ingress port onto dashboard and webhook listeners.

The external router authenticates twice: Sprites strips its account bearer at
its edge, and this process verifies the NAS replay secret before restoring the
original application's Authorization header. Bodies and redirects are not retried.
"""
from __future__ import annotations

import asyncio
import hmac
import os
from pathlib import Path
import re

import aiohttp
from aiohttp import web
from multidict import CIMultiDict
from yarl import URL

WAKE_MARKER = Path("/opt/data/state/sprites-wake")
DASHBOARD_PORT = 9119
WEBHOOK_PORT = 8644
WEBHOOK = re.compile(r'^/(?:p/[a-z0-9._-]+/)?webhooks/')
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade'}
SESSION = web.AppKey('session', aiohttp.ClientSession)
SECRET = web.AppKey('secret', str)


def forwarded_headers(headers):
    excluded = HOP_HEADERS | {h.strip().lower() for h in headers.get('Connection', '').split(',')}
    return CIMultiDict((key, value) for key, value in headers.items() if key.lower() not in excluded)


async def websocket(request, target, headers):
    protocols = [p.strip() for p in request.headers.get('Sec-WebSocket-Protocol', '').split(',') if p.strip()]
    for key in ('Sec-WebSocket-Key', 'Sec-WebSocket-Version', 'Sec-WebSocket-Extensions', 'Sec-WebSocket-Protocol'):
        headers.popall(key, None)
    async with request.app[SESSION].ws_connect(target, headers=headers, protocols=protocols, max_msg_size=16 * 1024 * 1024) as upstream:
        downstream = web.WebSocketResponse(protocols=[upstream.protocol] if upstream.protocol else (), max_msg_size=16 * 1024 * 1024)
        await downstream.prepare(request)

        async def pump(source, destination):
            async for message in source:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await destination.send_str(message.data)
                elif message.type == aiohttp.WSMsgType.BINARY:
                    await destination.send_bytes(message.data)
                elif message.type == aiohttp.WSMsgType.ERROR:
                    break
            await destination.close(code=source.close_code or 1000)

        tasks = [asyncio.create_task(pump(upstream, downstream)), asyncio.create_task(pump(downstream, upstream))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return downstream


async def proxy(request):
    secret = request.headers.get('X-Hermes-Ingress-Secret', '')
    if not hmac.compare_digest(secret, request.app[SECRET]):
        raise web.HTTPUnauthorized()
    if Path('/etc/hermes-sprites-state/stopped').exists():
        raise web.HTTPServiceUnavailable(text='Agent deliberately stopped')
    headers = forwarded_headers(request.headers)
    headers.popall('X-Hermes-Ingress-Secret', None)
    original_auth = headers.popall('X-Hermes-Original-Authorization', [])
    headers.popall('Authorization', None)
    if original_auth:
        headers['Authorization'] = original_auth[-1]
    # The external router sets this host; the native edge's host belongs to Sprites.
    host = headers.popall('X-Hermes-Original-Host', [])
    if len(host) != 1 or not re.fullmatch(r'[a-z0-9.-]+', host[0]):
        raise web.HTTPBadRequest()
    headers['Host'] = host[0]
    headers['X-Forwarded-Host'] = host[0]
    headers['X-Forwarded-Proto'] = 'https'
    headers['Fly-Replay-Src'] = 'state=' + secret
    headers.popall('Fly-Replay', None)
    # Control-plane status probes must not reset the idle clock or trigger relay reconnects.
    if request.path not in ('/api/status', '/health', '/healthz'):
        marker = WAKE_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    port = WEBHOOK_PORT if WEBHOOK.match(request.path) else DASHBOARD_PORT
    target = URL(f'http://127.0.0.1:{port}' + request.raw_path, encoded=True)
    try:
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            return await websocket(request, target, headers)
        async with request.app[SESSION].request(request.method, target, headers=headers,
                data=request.content.iter_chunked(64 * 1024), allow_redirects=False) as upstream:
            response = web.StreamResponse(status=upstream.status, headers=forwarded_headers(upstream.headers))
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response
    except (aiohttp.ClientError, asyncio.TimeoutError):
        # Never echo URLs, headers, credentials, or retry a partially sent webhook.
        raise web.HTTPBadGateway(text='Hermes service unavailable') from None


async def session_context(app):
    async with aiohttp.ClientSession(auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            skip_auto_headers={'Content-Type', 'User-Agent', 'Accept-Encoding'}) as session:
        app[SESSION] = session
        yield


def application(secret: str):
    if len(secret) < 32:
        raise ValueError('A replay secret is required')
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app[SECRET] = secret
    app.cleanup_ctx.append(session_context)
    app.router.add_route('*', '/{path:.*}', proxy)
    return app


if __name__ == '__main__':
    web.run_app(application(os.environ['AGENT_DASHBOARD_REPLAY_SECRET']), host='0.0.0.0', port=8080, access_log=None)
