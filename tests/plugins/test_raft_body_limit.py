from __future__ import annotations

import asyncio

import pytest

from plugins.platforms.raft.adapter import _read_limited_request_body


class _Content:
    def __init__(self, body: bytes):
        self.body = body
        self.read_size = None

    async def readexactly(self, size: int) -> bytes:
        self.read_size = size
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


class _Request:
    def __init__(self, body: bytes):
        self.content = _Content(body)


def test_read_limited_request_body_returns_short_body():
    request = _Request(b'{"ok":true}')

    body = asyncio.run(_read_limited_request_body(request, 16))

    assert body == b'{"ok":true}'
    assert request.content.read_size == 17


def test_read_limited_request_body_rejects_oversized_body():
    request = _Request(b"x" * 17)

    with pytest.raises(ValueError, match="payload_too_large"):
        asyncio.run(_read_limited_request_body(request, 16))

    assert request.content.read_size == 17
