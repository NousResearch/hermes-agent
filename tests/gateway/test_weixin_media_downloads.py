"""Weixin media GETs recover from a truncated CDN response without hiding permanent failures."""

import asyncio

import pytest

from gateway.platforms import weixin

aiohttp = pytest.importorskip("aiohttp")


class _Response:
    def __init__(self, outcome):
        self.outcome = outcome
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.closed = True

    def raise_for_status(self):
        if isinstance(self.outcome, aiohttp.ClientResponseError):
            raise self.outcome

    async def read(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Session:
    def __init__(self, *outcomes):
        self.responses = [_Response(outcome) for outcome in outcomes]
        self.calls = 0

    def get(self, _url):
        if self.calls:
            assert self.responses[self.calls - 1].closed
        result = self.responses[self.calls]
        self.calls += 1
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    aiohttp.ClientPayloadError("response ended before Content-Length"),
    aiohttp.ServerDisconnectedError("connection reset"),
    TimeoutError(),
])
async def test_transient_media_failure_restarts_the_get_and_uses_complete_response(monkeypatch, failure):
    monkeypatch.setattr(weixin, "MEDIA_DOWNLOAD_RETRY_DELAY_SECONDS", 0, raising=False)
    session = _Session(failure, b"complete encrypted payload")

    body = await weixin._download_bytes(session, url="https://cdn.example/media", timeout_seconds=2)

    assert body == b"complete encrypted payload"
    assert session.calls == 2
    assert all(response.closed for response in session.responses)


@pytest.mark.asyncio
async def test_truncated_http_body_restarts_with_a_fresh_request(monkeypatch):
    monkeypatch.setattr(weixin, "MEDIA_DOWNLOAD_RETRY_DELAY_SECONDS", 0, raising=False)
    payload = b"complete encrypted payload"
    calls = 0

    async def serve_media(reader, writer):
        nonlocal calls
        await reader.readuntil(b"\r\n\r\n")
        calls += 1
        body = payload[:5] if calls == 1 else payload
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve_media, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        async with aiohttp.ClientSession(trust_env=False) as session:
            body = await weixin._download_bytes(session, url=f"http://127.0.0.1:{port}/media", timeout_seconds=2)

    assert body == payload
    assert calls == 2


@pytest.mark.asyncio
async def test_repeated_truncation_fails_after_bounded_fresh_requests(monkeypatch):
    monkeypatch.setattr(weixin, "MEDIA_DOWNLOAD_RETRY_DELAY_SECONDS", 0, raising=False)
    session = _Session(*(aiohttp.ClientPayloadError("truncated") for _ in range(3)))

    with pytest.raises(aiohttp.ClientPayloadError):
        await weixin._download_bytes(session, url="https://cdn.example/media", timeout_seconds=2)

    assert session.calls == 3
    assert all(response.closed for response in session.responses)


@pytest.mark.asyncio
async def test_http_rejection_is_not_retried(monkeypatch):
    monkeypatch.setattr(weixin, "MEDIA_DOWNLOAD_RETRY_DELAY_SECONDS", 0, raising=False)
    rejection = aiohttp.ClientResponseError(None, (), status=403, message="Forbidden")
    session = _Session(rejection)

    with pytest.raises(aiohttp.ClientResponseError) as raised:
        await weixin._download_bytes(session, url="https://cdn.example/media", timeout_seconds=2)

    assert raised.value.status == 403
    assert session.calls == 1
    assert session.responses[0].closed
