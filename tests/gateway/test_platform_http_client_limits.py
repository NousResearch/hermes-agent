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

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)


def test_returns_none_when_httpx_unavailable(monkeypatch):
    """If httpx can't be imported, the helper returns None so callers
    fall back to httpx's built-in Limits default without raising."""
    import gateway.platforms._http_client_limits as mod
    monkeypatch.setattr(mod, "httpx", None)
    assert mod.platform_httpx_limits() is None


def test_default_limits_tighten_keepalive_below_httpx_default():
    import httpx
    from gateway.platforms._http_client_limits import platform_httpx_limits
    limits = platform_httpx_limits()
    assert isinstance(limits, httpx.Limits)
    # httpx default keepalive_expiry is 5.0 — ours must be shorter so
    # CLOSE_WAIT sockets drain promptly behind proxies like Warp.
    assert limits.keepalive_expiry is not None
    assert limits.keepalive_expiry < 5.0
    # max_keepalive_connections must be positive and reasonable for a
    # single adapter (platform APIs rarely parallelise beyond ~10).
    assert limits.max_keepalive_connections is not None
    assert 1 <= limits.max_keepalive_connections <= 50


def test_env_override_keepalive_expiry(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "7.5")
    from gateway.platforms._http_client_limits import platform_httpx_limits
    limits = platform_httpx_limits()
    assert limits.keepalive_expiry == 7.5


def test_env_override_max_keepalive(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "25")
    from gateway.platforms._http_client_limits import platform_httpx_limits
    limits = platform_httpx_limits()
    assert limits.max_keepalive_connections == 25


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


def test_helper_is_importable_from_every_platform_that_uses_it():
    """Every persistent-httpx-client platform adapter imports this helper.
    If any of those modules fails to import, this test surfaces it before
    the regression shows up as a runtime adapter-startup crash."""
    # Just importing exercises the helper's import path for each adapter.
    import gateway.platforms.qqbot.adapter  # noqa: F401
    import gateway.platforms.wecom  # noqa: F401
    import gateway.platforms.dingtalk  # noqa: F401
    import gateway.platforms.signal  # noqa: F401
    import gateway.platforms.bluebubbles  # noqa: F401
    import gateway.platforms.wecom_callback  # noqa: F401


class TestWhatsappTypingLeakFix:
    """#18451 — whatsapp.send_typing previously used a bare
    `await self._http_session.post(...)` which leaked the aiohttp
    response object until GC, holding its TCP socket in CLOSE_WAIT.
    Must now wrap the call in `async with` so the response is
    released immediately when the call returns.

    We verify by inspecting the source text rather than exercising
    the coroutine — the test suite would otherwise need a live
    aiohttp server, and the contract we care about is structural.
    """

    def test_bare_await_removed(self):
        import inspect
        import gateway.platforms.whatsapp as mod

        src = inspect.getsource(mod.WhatsAppAdapter.send_typing)
        # The fix must be structural: the post() call is inside an
        # `async with`, not a bare `await`.
        assert "async with self._http_session.post(" in src, (
            "send_typing must wrap self._http_session.post(...) in "
            "`async with` to release the aiohttp response socket "
            "(#18451). Otherwise the response sits in CLOSE_WAIT "
            "until GC."
        )
        # The old bare-await form must be gone.
        assert "await self._http_session.post(" not in src
