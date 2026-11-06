"""HTTP contract for fixed-token webhook auth and legacy downgrade boundaries."""
import asyncio
import base64
import hashlib
import hmac
import time
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from tests.gateway.fixtures.webhook_route_authority import mount_authority

SECRET = "route-secret"
BODY = b'{}'
V1 = hmac.new(SECRET.encode(), BODY, hashlib.sha256).hexdigest()
RECOGNIZED = ["svix-signature", "webhook-signature", "linear-signature", "X-Hub-Signature-256", "X-Gitlab-Token", "X-Webhook-Signature-V2", "X-Webhook-Signature"]


def make_app(routes=None, global_secret="global-secret", rate_limit=100):
    a = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "secret": global_secret, "rate_limit": rate_limit, "routes": routes or {"probe": {"secret": SECRET, "prompt": "hi"}}}))
    a.handle_message = AsyncMock()
    app = web.Application()
    mount_authority(app, a)
    app.router.add_post("/webhooks/{route_name}", a._handle_webhook)
    app.router.add_post("/p/{profile}/webhooks/{route_name}", a._handle_webhook)
    return a, app


@pytest.mark.asyncio
@pytest.mark.parametrize("route,token,status", [("scoped", SECRET, 202), ("scoped", "global-secret", 401), ("global", "global-secret", 202), ("global", SECRET, 401), ("blank", SECRET, 403), ("missing", SECRET, 403)])
async def test_effective_secret_and_no_dispatch(route, token, status):
    routes = {"scoped": {"secret": SECRET}, "global": {"prompt": "x"}, "blank": {"secret": "  "}, "missing": {"secret": ""}}
    a, app = make_app(routes)
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/' + route, data=BODY, headers={"Authorization": "Bearer " + token})
        assert r.status == status
        assert a.handle_message.await_count == (1 if status == 202 else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization,status", [("Bearer route-secret", 202), ("bEaReR route-secret", 202), ("Bearer wrong", 401), ("Bearer", 401), ("Bearer ", 401), ("Basic route-secret", 401), ("", 401), ("Bearer tökén", 401)])
async def test_bearer_scheme(authorization, status):
    a, app = make_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/probe', data=BODY, headers={"Authorization": authorization})
        assert r.status == status
        assert a.handle_message.await_count == (status == 202)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", RECOGNIZED)
@pytest.mark.parametrize("value", ["", "invalid"])
async def test_recognized_signature_blocks_bearer(name, value):
    a, app = make_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/probe', data=BODY, headers={name: value, "Authorization": "Bearer " + SECRET})
        assert r.status == 401
        a.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", RECOGNIZED[:-1])
@pytest.mark.parametrize("bearer", [False, True])
async def test_empty_preexisting_header_still_allows_valid_v1(name, bearer):
    a, app = make_app()
    headers = {name: "", "X-Webhook-Signature": V1}
    if bearer:
        headers["Authorization"] = "Bearer " + SECRET
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/probe', data=BODY, headers=headers)
        assert r.status == 202
        assert a.handle_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp", [None, "invalid", "1", "fresh"])
async def test_nonempty_v2_never_downgrades(timestamp):
    a, app = make_app()
    headers = {"X-Webhook-Signature-V2": "invalid", "X-Webhook-Signature": V1, "Authorization": "Bearer " + SECRET}
    if timestamp is not None:
        headers["X-Webhook-Timestamp"] = str(int(time.time())) if timestamp == "fresh" else timestamp
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/probe', data=BODY, headers=headers)
        assert r.status == 401
        a.handle_message.assert_not_awaited()


def valid_headers(kind):
    timestamp = str(int(time.time()))
    if kind == "v2":
        sig = hmac.new(SECRET.encode(), timestamp.encode() + b'.' + BODY, hashlib.sha256).hexdigest()
        return {"X-Webhook-Timestamp": timestamp, "X-Webhook-Signature-V2": sig}
    if kind == "v1": return {"X-Webhook-Signature": V1}
    if kind == "github": return {"X-Hub-Signature-256": "sha256=" + V1}
    if kind == "linear": return {"linear-signature": V1}
    if kind == "gitlab": return {"webhook-id": "id", "webhook-timestamp": timestamp, "X-Gitlab-Token": SECRET}
    sig = base64.b64encode(hmac.new(SECRET.encode(), b'id.' + timestamp.encode() + b'.' + BODY, hashlib.sha256).digest()).decode()
    prefix = "svix" if kind == "svix" else "webhook"
    return {prefix + "-id": "id", prefix + "-timestamp": timestamp, prefix + "-signature": "v1," + sig}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["v2", "v1", "github", "linear", "gitlab", "svix", "standard"])
async def test_valid_preexisting_schemes(kind):
    a, app = make_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/probe', data=BODY, headers=valid_headers(kind))
        assert r.status == 202
        assert a.handle_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["svix-id", "svix-timestamp", "svix-signature"])
async def test_partial_svix_blocks_gitlab_and_bearer(name):
    a, app = make_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/webhooks/probe', data=BODY, headers={name: "partial", "X-Gitlab-Token": SECRET, "Authorization": "Bearer " + SECRET})
        assert r.status == 401
        a.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [b'Authorization: Bearer \xff', b'X-Gitlab-Token: \xff'])
async def test_raw_malformed_header_rejected(header):
    a, app = make_app()
    server = TestServer(app)
    await server.start_server()
    try:
        reader, writer = await asyncio.open_connection(server.host, server.port)
        try:
            writer.write(b'POST /webhooks/probe HTTP/1.1\r\nHost: localhost\r\n' + header + b'\r\nContent-Length: 2\r\nConnection: close\r\n\r\n' + BODY)
            await writer.drain()
            assert (await reader.read()).split(b'\r\n', 1)[0] == b'HTTP/1.1 401 Unauthorized'
        finally:
            writer.close()
            await writer.wait_closed()
        a.handle_message.assert_not_awaited()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_bad_bearer_does_not_consume_rate_limit():
    a, app = make_app(rate_limit=2)
    async with TestClient(TestServer(app)) as client:
        for _ in range(3):
            r = await client.post('/webhooks/probe', data=BODY, headers={"Authorization": "Bearer wrong"})
            assert r.status == 401
        assert a.handle_message.await_count == 0
        r = await client.post('/webhooks/probe', data=BODY, headers={"Authorization": "Bearer " + SECRET})
        assert r.status == 202
        assert a.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_bearer_cannot_cross_profile_binding():
    a, app = make_app(routes={"probe": {"secret": SECRET, "profile": "secondary"}})
    async with TestClient(TestServer(app)) as client:
        for path in ('/webhooks/probe', '/p/unknown/webhooks/probe'):
            r = await client.post(path, data=BODY, headers={"Authorization": "Bearer " + SECRET})
            assert r.status == 404
        a.handle_message.assert_not_awaited()
