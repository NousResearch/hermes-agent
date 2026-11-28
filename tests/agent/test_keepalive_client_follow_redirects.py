"""Regression: the keepalive httpx client must FOLLOW redirects.

Providers legitimately answer a valid request with an HTTP 307 pointing at
their canonical endpoint:

  * DeepSeek's CloudFront answers plain ``http://api.deepseek.com/v1`` with a
    307 to the https URL (see issue #39765).
  * Gateways that re-sign request paths answer ``POST /v1/chat/completions``
    with a 307 to ``/r/v1/chat/completions?token=...``.

httpx does NOT follow redirects by default (unlike requests), so the OpenAI
SDK surfaced the 307 as a hard APIStatusError and every call failed.

This is safe with respect to credentials: httpx strips the ``Authorization``
header when a redirect leaves the origin, and keeps it for a same-origin or
http->https redirect (see httpx/_client.py::_redirect_headers). The two cases
that must still be followed are therefore exactly the two that keep the
credential attached to the same provider.

Covers both construction paths in build_keepalive_http_client: the direct
(no-proxy) path and the proxy path.
"""

import httpx
import pytest

from agent.process_bootstrap import build_keepalive_http_client


@pytest.mark.parametrize("async_mode", [False, True])
def test_keepalive_client_follows_redirects(async_mode, monkeypatch):
    """The built client must have follow_redirects enabled."""
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    client = build_keepalive_http_client("https://provider.example.com/v1",
                                         async_mode=async_mode)
    assert client is not None
    assert client.follow_redirects is True
    if async_mode:
        # Async clients are never shared; close the coroutine cleanly.
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            client.aclose()
        )
    else:
        client.close()


def test_keepalive_client_follows_redirect_end_to_end(monkeypatch):
    """A 307 must be followed to the canonical URL, not surfaced as an error.

    Uses a local transport stub so no network access is required.
    """
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                307,
                headers={"Location": "https://provider.example.com/r/v1/chat/completions"},
            )
        return httpx.Response(200, json={"ok": True})

    client = build_keepalive_http_client("https://provider.example.com/v1")
    assert client is not None
    client._mounts = {}
    client._transport = httpx.MockTransport(handler)
    try:
        r = client.post(
            "https://provider.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "x"},
        )
    finally:
        client.close()

    assert r.status_code == 200
    assert seen["path"] == "/r/v1/chat/completions"
    # Same-origin redirect keeps the credential (httpx strips it only on a
    # cross-origin redirect), so the provider still receives the API key.
    assert seen["auth"] == "Bearer test-key"


def test_keepalive_client_still_strips_auth_on_cross_origin_redirect(monkeypatch):
    """A redirect to another origin must NOT carry the API key."""
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["auth"] = request.headers.get("authorization")
        if request.url.host == "provider.example.com":
            return httpx.Response(
                307, headers={"Location": "https://other.example.com/v1/chat/completions"}
            )
        return httpx.Response(200, json={"ok": True})

    client = build_keepalive_http_client("https://provider.example.com/v1")
    assert client is not None
    client._mounts = {}
    client._transport = httpx.MockTransport(handler)
    try:
        r = client.post(
            "https://provider.example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "x"},
        )
    finally:
        client.close()

    assert r.status_code == 200
    assert seen["host"] == "other.example.com"
    assert seen["auth"] is None
