from __future__ import annotations

import httpx
import pytest

from agent.httpx_bounded_response import (
    HTTPXResponseBodyTooLarge,
    HTTPXUnsupportedContentEncoding,
    read_httpx_response_bytes_limited,
)


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks
        self.yielded = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self):
        pass


def _response(stream: _TrackedStream, **headers: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers=headers,
        request=httpx.Request("GET", "https://provider.invalid/data"),
        stream=stream,
    )


@pytest.mark.asyncio
async def test_reads_complete_identity_response():
    stream = _TrackedStream(b'{"ok":', b"true}")
    response = _response(
        stream,
        **{"Content-Encoding": "identity", "Content-Length": "11"},
    )

    assert await read_httpx_response_bytes_limited(response, max_bytes=11) == (
        b'{"ok":true}'
    )
    assert stream.yielded == 2


@pytest.mark.asyncio
async def test_stops_when_stream_crosses_limit():
    stream = _TrackedStream(b"abcd", b"efgh", b"unread")
    response = _response(stream)

    with pytest.raises(HTTPXResponseBodyTooLarge, match="exceeds 6 bytes"):
        await read_httpx_response_bytes_limited(response, max_bytes=6)

    assert stream.yielded == 2


@pytest.mark.asyncio
async def test_rejects_declared_oversize_without_reading_body():
    stream = _TrackedStream(b"unread")
    response = _response(stream, **{"Content-Length": "7"})

    with pytest.raises(HTTPXResponseBodyTooLarge, match="exceeds 6 bytes"):
        await read_httpx_response_bytes_limited(response, max_bytes=6)

    assert stream.yielded == 0


@pytest.mark.asyncio
async def test_rejects_encoded_response_without_decoding_body():
    stream = _TrackedStream(b"compressed bytes stay unread")
    response = _response(stream, **{"Content-Encoding": "gzip"})

    with pytest.raises(
        HTTPXUnsupportedContentEncoding,
        match="unsupported Content-Encoding: gzip",
    ):
        await read_httpx_response_bytes_limited(response, max_bytes=1024)

    assert stream.yielded == 0
