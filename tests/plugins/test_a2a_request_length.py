"""The A2A POST handler must reject a negative ``Content-Length``. ``rfile.read(-1)`` reads until the
client closes the connection, so a negative value would skip the ``_MAX_BODY`` cap entirely."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest

from plugins.platforms.a2a.adapter import _MAX_BODY
from tests.plugins.test_a2a_plugin import _make_live_adapter, _send_body


def _raw_post(port: int, body: bytes, content_length: str) -> int:
    """POST ``body`` with the given ``Content-Length`` header, close the write side, return the status."""
    head = (
        "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer topsecret\r\n"
        f"Content-Type: application/json\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
    ).encode()
    with socket.create_connection(("127.0.0.1", port), timeout=15) as sock:
        sock.sendall(head + body)
        sock.shutdown(socket.SHUT_WR)
        response = b""
        while chunk := sock.recv(65536):
            response += chunk
    return int(response.split(b" ", 2)[1])


@pytest.mark.parametrize("oversized", [False, True])
def test_negative_content_length_is_rejected(monkeypatch, oversized):
    monkeypatch.setenv("A2A_BEARER_TOKEN", "topsecret")
    monkeypatch.setenv("A2A_HOST", "127.0.0.1")
    adapter, _ = _make_live_adapter(monkeypatch)
    body = json.dumps(_send_body("hello")).encode()
    if oversized:
        body += b" " * (_MAX_BODY + 1)

    async def run():
        assert await adapter.connect() is True
        try:
            return await asyncio.to_thread(_raw_post, adapter.port, body, "-1")
        finally:
            await adapter.disconnect()

    assert asyncio.run(run()) == 400


def test_negative_content_length_rejected_before_body_eof(monkeypatch):
    """A malformed length must get HTTP 400 without reading body bytes or waiting for EOF."""
    monkeypatch.setenv("A2A_BEARER_TOKEN", "topsecret")
    monkeypatch.setenv("A2A_HOST", "127.0.0.1")
    adapter, _ = _make_live_adapter(monkeypatch)

    def headers_only() -> int:
        headers = (
            "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer topsecret\r\n"
            "Content-Type: application/json\r\nContent-Length: -1\r\nConnection: close\r\n\r\n"
        ).encode("ascii")
        with socket.create_connection(("127.0.0.1", adapter.port), timeout=10) as sock:
            sock.settimeout(5)
            sock.sendall(headers)
            # Keep the write side open: rfile.read(-1) would wait for EOF.
            with sock.makefile("rb") as response:
                return int(response.readline().split(b" ", 2)[1])

    async def run():
        assert await adapter.connect() is True
        try:
            return await asyncio.to_thread(headers_only)
        finally:
            await adapter.disconnect()

    assert asyncio.run(run()) == 400
