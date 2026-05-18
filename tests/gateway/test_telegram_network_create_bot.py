"""Tests for ``gateway.platforms.telegram_network.create_bot_with_fallback``.

Covers issue #20915: the one-shot ``send_message`` tool path was creating
a plain ``telegram.Bot`` that bypasses :class:`TelegramFallbackTransport`,
so it timed out on networks where ``api.telegram.org``'s system-DNS IP is
blocked.  The new helper mirrors the gateway adapter's network setup.

The test-suite mock for the ``telegram`` module replaces ``Bot`` and
``HTTPXRequest`` with ``MagicMock``s, so we assert on the recorded call
args (not on an introspected ``Bot.request`` instance).
"""

import sys

import pytest

from gateway.platforms import telegram_network as tnet


@pytest.fixture
def isolated_env(monkeypatch):
    """Clear all env vars the helper consults."""
    for key in [
        "HERMES_TELEGRAM_DISABLE_FALLBACK_IPS",
        "HERMES_TELEGRAM_FALLBACK_IPS",
        "TELEGRAM_PROXY",
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
        "https_proxy", "http_proxy", "all_proxy",
        "NO_PROXY", "no_proxy",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def telegram_mocks(monkeypatch):
    """Reset ``Bot`` / ``HTTPXRequest`` mock call records for a clean assertion.

    The session-scoped mock in ``tests/gateway/conftest.py`` keeps
    ``telegram`` and ``telegram.request`` as ``MagicMock``s; we just
    reset their call history so each test sees a clean slate.
    """
    bot_cls = sys.modules["telegram"].Bot
    httpx_request_cls = sys.modules["telegram.request"].HTTPXRequest
    bot_cls.reset_mock()
    httpx_request_cls.reset_mock()
    return bot_cls, httpx_request_cls


class TestCreateBotWithFallback:
    """Behaviour matrix for create_bot_with_fallback."""

    @pytest.mark.asyncio
    async def test_disabled_env_skips_discovery_and_returns_plain_bot(
        self, isolated_env, telegram_mocks, monkeypatch
    ):
        bot_cls, httpx_request_cls = telegram_mocks
        monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1")

        called = {"discover": False}

        async def _no_discover():
            called["discover"] = True
            return []

        monkeypatch.setattr(tnet, "discover_fallback_ips", _no_discover)

        await tnet.create_bot_with_fallback("123:abc")

        assert called["discover"] is False
        bot_cls.assert_called_once_with(token="123:abc")
        httpx_request_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_env_ips_skip_discovery_and_use_fallback_transport(
        self, isolated_env, telegram_mocks, monkeypatch
    ):
        bot_cls, httpx_request_cls = telegram_mocks
        monkeypatch.setenv(
            "HERMES_TELEGRAM_FALLBACK_IPS", "149.154.167.220, 149.154.165.120"
        )

        called = {"discover": False}

        async def _no_discover():
            called["discover"] = True
            return []

        monkeypatch.setattr(tnet, "discover_fallback_ips", _no_discover)

        await tnet.create_bot_with_fallback("123:abc")

        assert called["discover"] is False
        httpx_request_cls.assert_called_once()
        kwargs = httpx_request_cls.call_args.kwargs
        transport = kwargs["httpx_kwargs"]["transport"]
        assert isinstance(transport, tnet.TelegramFallbackTransport)
        bot_cls.assert_called_once()
        assert bot_cls.call_args.kwargs["token"] == "123:abc"
        assert "request" in bot_cls.call_args.kwargs

    @pytest.mark.asyncio
    async def test_discovery_used_when_no_env_and_returns_ips(
        self, isolated_env, telegram_mocks, monkeypatch
    ):
        bot_cls, httpx_request_cls = telegram_mocks

        async def _fake_discover():
            return ["149.154.167.220"]

        monkeypatch.setattr(tnet, "discover_fallback_ips", _fake_discover)

        await tnet.create_bot_with_fallback("123:abc")

        httpx_request_cls.assert_called_once()
        bot_cls.assert_called_once()
        assert "request" in bot_cls.call_args.kwargs

    @pytest.mark.asyncio
    async def test_no_fallback_no_proxy_returns_plain_bot(
        self, isolated_env, telegram_mocks, monkeypatch
    ):
        bot_cls, httpx_request_cls = telegram_mocks

        async def _empty_discover():
            return []

        monkeypatch.setattr(tnet, "discover_fallback_ips", _empty_discover)

        await tnet.create_bot_with_fallback("123:abc")

        bot_cls.assert_called_once_with(token="123:abc")
        httpx_request_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_failure_falls_back_to_plain_bot(
        self, isolated_env, telegram_mocks, monkeypatch
    ):
        bot_cls, httpx_request_cls = telegram_mocks

        async def _broken_discover():
            raise RuntimeError("dns over https unavailable")

        monkeypatch.setattr(tnet, "discover_fallback_ips", _broken_discover)

        await tnet.create_bot_with_fallback("123:abc")

        bot_cls.assert_called_once_with(token="123:abc")
        httpx_request_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_takes_precedence_over_fallback(
        self, isolated_env, telegram_mocks, monkeypatch
    ):
        bot_cls, httpx_request_cls = telegram_mocks
        monkeypatch.setenv("TELEGRAM_PROXY", "http://proxy.example:3128")
        monkeypatch.setenv(
            "HERMES_TELEGRAM_FALLBACK_IPS", "149.154.167.220"
        )

        await tnet.create_bot_with_fallback("123:abc")

        httpx_request_cls.assert_called_once()
        kwargs = httpx_request_cls.call_args.kwargs
        # Proxy path uses the ``proxy=`` kwarg, not the fallback transport
        assert "proxy" in kwargs
        assert kwargs["proxy"]
        assert "httpx_kwargs" not in kwargs
