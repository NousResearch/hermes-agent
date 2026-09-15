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


def test_returns_none_when_httpx_unavailable(monkeypatch):
    """If httpx can't be imported, the helper returns None so callers
    fall back to httpx's built-in Limits default without raising."""
    import gateway.platforms._http_client_limits as mod
    monkeypatch.setattr(mod, "httpx", None)
    assert mod.platform_httpx_limits() is None


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
        import plugins.platforms.whatsapp.adapter as mod

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
