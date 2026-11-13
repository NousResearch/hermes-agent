"""DoH fallback-IP discovery never buffers an oversized provider body.

A captive portal, proxy or broken DoH endpoint can answer with an arbitrarily large body. That
provider is treated as failed, and the other provider's answer is still used.
"""

import json

import httpx
import pytest

import hermes_agent.plugins.platforms.telegram.telegram_network as tnet


class _Stream(httpx.AsyncByteStream):
    def __init__(self, *chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def _answer(ip):
    """A network-shaped (streamed, not pre-read) DoH answer."""
    return _Stream(json.dumps({"Answer": [{"type": 1, "data": ip}]}).encode())


class _ChunkedBody(httpx.AsyncByteStream):
    """Streams ``total`` bytes of a JSON body, counting how much the reader pulled."""

    def __init__(self, total):
        self.total, self.sent = total, 0

    async def __aiter__(self):
        head = b'{"Answer":[{"type":1,"data":"149.154.167.99"}],"pad":"'
        self.sent += len(head)
        yield head
        while self.sent < self.total:
            self.sent += 16384
            yield b"x" * 16384
        yield b'"}'


@pytest.mark.asyncio
async def test_oversized_provider_is_ignored_and_never_fully_read(monkeypatch):
    big = _ChunkedBody(64 * 1024 * 1024)

    def handler(request):
        if request.url.host == "dns.google":
            return httpx.Response(200, stream=big)
        return httpx.Response(200, stream=_answer("149.154.167.51"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(tnet.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), timeout=kw.get("timeout")))
    monkeypatch.setattr(tnet, "_resolve_system_dns", lambda: set())

    ips = await tnet.discover_fallback_ips()

    assert "149.154.167.99" not in ips  # the oversized answer was dropped, not parsed
    assert ips[0] == "149.154.167.51"   # the healthy provider still drives discovery
    assert big.sent < 2 * tnet._DOH_MAX_BODY_BYTES


@pytest.mark.asyncio
async def test_normal_answers_parse_and_request_identity_encoding(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers.get("accept-encoding"))
        return httpx.Response(200, stream=_answer("149.154.167.51" if request.url.host == "dns.google" else "149.154.167.52"))

    real_client = httpx.AsyncClient
    monkeypatch.setattr(tnet.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), timeout=kw.get("timeout")))
    monkeypatch.setattr(tnet, "_resolve_system_dns", lambda: set())

    ips = await tnet.discover_fallback_ips()

    assert ips[:2] == ["149.154.167.51", "149.154.167.52"]
    assert seen == ["identity", "identity"]
