"""Tests for the shared httpx.Limits helper that all long-lived platform
adapters use to tighten their keep-alive pool.

Context: #18451 — on macOS behind Cloudflare Warp, httpx's default
keepalive_expiry=5s let idle CLOSE_WAIT sockets accumulate across
multiple long-lived gateway adapters (QQ Bot, Feishu, WeCom, DingTalk,
Signal, BlueBubbles, WeCom-callback) until the process hit the default
256 fd limit.  These tests just verify the helper returns sensibly
tuned limits and respects env-var overrides; the actual fd-pressure
behaviour is only observable at runtime under load.
"""

from __future__ import annotations

import pytest

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)

def test_env_override_rejects_garbage(monkeypatch):
    """Malformed env values fall back to defaults rather than raising."""
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "not-a-number")
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "-3")
    from gateway.platforms._http_client_limits import platform_httpx_limits
    limits = platform_httpx_limits()
    # Non-positive / non-numeric → fell back to defaults (not the override values)
    assert limits.keepalive_expiry is not None and limits.keepalive_expiry > 0
    assert limits.max_keepalive_connections is not None
    assert limits.max_keepalive_connections > 0


@pytest.mark.asyncio
async def test_platform_http_client_is_ssrf_guarded_with_shared_limits():
    """The public plugin client refuses private targets at connect time, re-checks redirects ahead of
    caller hooks, and uses the shared keepalive limits."""
    from gateway.platforms._http_client_limits import platform_httpx_limits
    from gateway.platforms.base import ssrf_redirect_guard
    from gateway.platforms.helpers import platform_http_client

    async def caller_hook(response):
        return None

    client = platform_http_client(event_hooks={"response": [caller_hook]})
    try:
        assert client.event_hooks["response"] == [ssrf_redirect_guard, caller_hook]
        pool = client._transport._pool
        limits = platform_httpx_limits()
        assert (pool._max_keepalive_connections, pool._keepalive_expiry) == (
            limits.max_keepalive_connections, limits.keepalive_expiry)
        with pytest.raises(ValueError):
            await client.get("http://169.254.169.254/latest/meta-data/")
    finally:
        await client.aclose()
