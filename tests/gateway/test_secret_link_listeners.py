"""The secret-link routes ride the gateway's existing listeners (api_server, webhook) — no port of
their own — and the api_server CORS allowlist does not block the page's same-origin form POST."""

import socket
from urllib.parse import urlencode

import aiohttp
import pytest

from gateway.config import PlatformConfig
from tests.gateway.secret_link_helpers import (
    BASE, SENTINEL, assert_secret_headers, clock_fixture, env_text, form_nonce, hermes_root_fixture, new_link, restarts_fixture,
    store_fixture)

pytestmark = pytest.mark.asyncio


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _api_server(port):
    from gateway.platforms.api_server import APIServerAdapter
    return APIServerAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": port, "key": "sk-test-strong-key-0123456789"}))


def _webhook(port):
    from gateway.platforms.webhook import WebhookAdapter
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": port, "routes": {}}))


@pytest.mark.parametrize("make_adapter", [_api_server, _webhook], ids=["api_server", "webhook"])
async def test_listener_serves_the_form_and_accepts_the_same_origin_post(
        hermes_root, store, restarts, make_adapter):
    port = _free_port()
    adapter = make_adapter(port)
    assert await adapter.connect() is True
    try:
        token = new_link(store)
        url = f"http://127.0.0.1:{port}/secret-link/{token}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                html = await resp.text()
                assert resp.status == 200
                assert resp.headers["Content-Security-Policy"].startswith("default-src 'none'")
                assert resp.headers["Strict-Transport-Security"] == "max-age=31536000"
            body = urlencode([("nonce", form_nonce(html)), ("key", "LISTENER_KEY"), ("secret", SENTINEL),
                              ("profile", "alpha")])
            async with session.post(url, data=body, headers={
                    "Origin": BASE, "Content-Type": "application/x-www-form-urlencoded"}) as resp:
                assert resp.status == 200
                assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await adapter.disconnect()
    assert f"LISTENER_KEY={SENTINEL}" in env_text(hermes_root / "profiles" / "alpha")


@pytest.mark.parametrize("make_adapter", [_api_server, _webhook], ids=["api_server", "webhook"])
async def test_listener_answers_other_methods_with_a_405_that_has_the_headers(
        hermes_root, store, make_adapter):
    port = _free_port()
    adapter = make_adapter(port)
    assert await adapter.connect() is True
    try:
        url = f"http://127.0.0.1:{port}/secret-link/{new_link(store)}"
        async with aiohttp.ClientSession() as session:
            for method in ("PUT", "DELETE", "OPTIONS"):
                # api_server's CORS middleware refuses OPTIONS (and a foreign Origin) with its own
                # 403 before the router answers; that refusal must carry the headers too.
                for headers in ({}, {"Origin": "https://evil.example"}):
                    async with session.request(method, url, headers=headers) as resp:
                        assert resp.status in (403, 405), (method, headers)
                        if method != "OPTIONS" and not headers:
                            assert resp.status == 405
                        assert_secret_headers(resp)
    finally:
        await adapter.disconnect()


async def test_api_server_body_cap_does_not_replace_the_oversized_form_400(hermes_root, store, monkeypatch):
    """AC-15: a form body over 16 KiB is a 400 page, even when it is also over the listener's
    own (larger) request cap, which would otherwise answer 413 without the headers."""
    from gateway.platforms import api_server

    monkeypatch.setattr(api_server, "MAX_REQUEST_BYTES", 20_000)
    port = _free_port()
    adapter = _api_server(port)
    assert await adapter.connect() is True
    try:
        url = f"http://127.0.0.1:{port}/secret-link/{new_link(store)}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=b"a=" + b"b" * 32_000, headers={
                    "Origin": BASE, "Content-Type": "application/x-www-form-urlencoded"}) as resp:
                assert resp.status == 400
                assert_secret_headers(resp)
    finally:
        await adapter.disconnect()
