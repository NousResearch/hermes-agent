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

from unittest.mock import AsyncMock, MagicMock

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


class _TrackingResponseCM:
    """Mimics aiohttp's ``_RequestContextManager``: the object returned by
    ``session.post(...)`` is BOTH an async context manager and directly
    awaitable, so it can be misused either way:

    - ``async with session.post(...) as resp:`` calls ``__aenter__`` then
      ``__aexit__`` — aiohttp releases the response (and its socket) in
      ``__aexit__``. This is the correct, leak-free usage.
    - ``await session.post(...)`` (a bare await, the #18451 bug) resolves
      the awaitable directly and returns the response with neither
      ``__aenter__`` nor ``__aexit__`` ever called — nothing releases the
      response, so its socket sits in CLOSE_WAIT until GC finalizes it.

    Recording which path was actually exercised lets the test assert the
    real behavioural contract (prompt release) instead of reading source.
    """

    def __init__(self, tracker):
        self._tracker = tracker
        self.response = MagicMock(status=200)

    async def __aenter__(self):
        self._tracker["entered"] = True
        return self.response

    async def __aexit__(self, *exc_info):
        self._tracker["exited"] = True
        return False

    def __await__(self):
        async def _bare_await():
            self._tracker["bare_awaited"] = True
            return self.response
        return _bare_await().__await__()


@pytest.fixture
def whatsapp_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=False)
    adapter._http_session = MagicMock()
    return adapter


class TestWhatsappTypingLeakFix:
    """#18451 — WhatsApp typing-indicator requests previously used a bare
    ``await self._http_session.post(...)``, which leaked the aiohttp
    response object until GC, holding its TCP socket in CLOSE_WAIT.

    A prior version of this test asserted the fix by reading
    ``inspect.getsource(...)`` for a literal ``async with`` substring; that
    version was removed in 61fa0f47ec ("purge low-value tests") as a
    source-reading anti-pattern (see AGENTS.md "Never read source code in
    tests" — it breaks on any correct refactor, e.g. two methods delegating
    to a shared helper, without the underlying release behaviour changing).

    This version restores coverage for the real #18451 regression
    behaviourally instead: it swaps in a fake response object that is BOTH
    an async context manager and a bare awaitable, and asserts which
    protocol the adapter actually invoked — the release-relevant fact —
    rather than how the call is spelled in source.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method_name", ["send_typing", "stop_typing"])
    async def test_typing_request_releases_response_via_context_manager(
        self, whatsapp_adapter, method_name
    ):
        from gateway.platforms.base import BasePlatformAdapter

        method = getattr(whatsapp_adapter, method_name, None)
        # ``getattr`` alone isn't enough: BasePlatformAdapter defines
        # send_typing/stop_typing as no-op stubs, so an adapter that hasn't
        # overridden one still resolves the attribute. Skip only when the
        # concrete WhatsApp adapter has no override of its own.
        overridden = method_name in type(whatsapp_adapter).__dict__ or any(
            method_name in klass.__dict__
            for klass in type(whatsapp_adapter).__mro__
            if klass not in (BasePlatformAdapter, object)
        )
        if method is None or not overridden:
            pytest.skip(f"WhatsAppAdapter has no {method_name} override")

        tracker: dict = {}
        whatsapp_adapter._http_session.post = MagicMock(
            return_value=_TrackingResponseCM(tracker)
        )

        await method("15551234567")

        assert tracker.get("entered") and tracker.get("exited"), (
            f"{method_name} must use `async with self._http_session.post(...)` "
            "so the aiohttp response is released immediately via __aexit__ "
            "(#18451). A bare `await` returns the response without releasing "
            "it, leaving its socket in CLOSE_WAIT until GC."
        )
        assert not tracker.get("bare_awaited"), (
            f"{method_name} appears to bare-await the response instead of "
            "entering it as an async context manager (#18451 regression)."
        )
