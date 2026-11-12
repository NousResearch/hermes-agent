"""Outbound URL media fetched over aiohttp never follows a redirect to a private address.

aiohttp follows redirects by default, so an adapter that only pre-checks the first URL lets a
public URL 302 to loopback/metadata and uploads the response into the chat.
"""
import asyncio
import socket
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.abc import AbstractResolver

from gateway.config import PlatformConfig
from tools import url_safety

PUBLIC_HOST = "redirector.test"
WEIXIN_CDN_HOST = "novac2c.cdn.weixin.qq.com"
INTERNAL_BODY = b"internal-only-body"


class _LoopbackResolver(AbstractResolver):
    """Dial the public-looking hostname on loopback, where the test's redirector listens."""

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return [{"hostname": host, "host": "127.0.0.1", "port": port,
                 "family": socket.AF_INET, "proto": 0, "flags": 0}]

    async def close(self):
        pass


async def _mattermost_fetch(session, url):
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t", extra={"url": "http://127.0.0.1:9"}))
    adapter._session, uploaded = session, []

    async def upload(channel_id, data, filename, content_type):
        uploaded.append(data)
        return "file-id"

    async def post_with_file(*args, **kwargs):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="post-id")

    async def send(*args, **kwargs):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="text-fallback")
    adapter._upload_file, adapter._post_with_file, adapter.send = upload, post_with_file, send
    await adapter.send_image("channel", url)
    loaded = await adapter._load_batch_image(url, 0)
    return uploaded + ([loaded[0]] if loaded else [])


async def _weixin_fetch(session, url):
    from plugins.platforms.weixin.adapter import WeixinAdapter
    adapter = WeixinAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._send_session = session
    try:
        path = await adapter._download_remote_media(url)
    except ValueError:
        return []
    return [await asyncio.to_thread(Path(path).read_bytes)]


async def _weixin_cdn_fetch(session, url):
    """Inbound media: ``full_url`` passes the CDN host allowlist, then the CDN answers 302."""
    from plugins.platforms.weixin import adapter as weixin
    try:
        return [await weixin._download_and_decrypt_media(
            session, cdn_base_url=weixin.WEIXIN_CDN_BASE_URL, encrypted_query_param=None, aes_key_b64=None,
            full_url=url, timeout_seconds=30)]
    except ValueError:
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize(("fetch", "public_host"), [
    (_mattermost_fetch, PUBLIC_HOST), (_weixin_fetch, PUBLIC_HOST), (_weixin_cdn_fetch, WEIXIN_CDN_HOST),
], ids=["mattermost", "weixin", "weixin-cdn"])
async def test_redirect_to_private_address_is_not_followed(fetch, public_host, monkeypatch):
    internal_hits = []

    async def internal(_request):
        internal_hits.append(1)
        return web.Response(body=INTERNAL_BODY, content_type="image/png")

    async def redirector(_request):
        raise web.HTTPFound(f"http://127.0.0.1:{internal_port}/")

    runners = []
    ports = []
    for handler in (internal, redirector):
        app = web.Application()
        app.router.add_get("/{tail:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        ports.append(site._server.sockets[0].getsockname()[1])
    internal_port, redirector_port = ports

    real_getaddrinfo = url_safety._getaddrinfo

    def getaddrinfo(host, port=None):
        if host == public_host:  # the first hop looks public to the pre-flight check
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))]
        return real_getaddrinfo(host, port)
    monkeypatch.setattr(url_safety, "_getaddrinfo", getaddrinfo)

    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(resolver=_LoopbackResolver()))
    try:
        fetched = await fetch(session, f"http://{public_host}:{redirector_port}/image.png")
    finally:
        await session.close()
        for runner in runners:
            await runner.cleanup()

    assert INTERNAL_BODY not in fetched
    assert internal_hits == []


@pytest.mark.asyncio
async def test_timeout_total_covers_the_whole_redirect_chain(monkeypatch):
    """aiohttp's own redirect loop spends one ``total``; following hops by hand must not reset it."""
    async def slow_hop(request):
        await asyncio.sleep(0.6)  # health: allow HX013 -- per-hop latency IS the behaviour: the chain must outlast one total
        if request.path == "/first":
            raise web.HTTPFound("/second")
        return web.Response(body=b"done")

    app = web.Application()
    app.router.add_get("/{tail:.*}", slow_hop)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    async def allow_all(_url):
        return True
    monkeypatch.setattr(url_safety, "async_is_safe_url", allow_all)

    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(asyncio.TimeoutError):  # each hop alone fits in 1 s, the chain does not
                async with url_safety.ssrf_checked_aiohttp_get(
                        session, f"http://127.0.0.1:{port}/first", timeout=aiohttp.ClientTimeout(total=1.0)) as resp:
                    await resp.read()
    finally:
        await runner.cleanup()
