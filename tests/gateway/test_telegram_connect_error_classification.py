"""Local-bind-exhaustion classification for the Telegram fallback transport (#107880).

Background
----------
The fallback transport walks Telegram-owned IPv4 literals when the hostname path
fails.  ``EADDRNOTAVAIL`` is not a remote-reachability failure — it means the local
ephemeral range is exhausted (macOS TIME_WAIT pile-up after ~2 days of getUpdates
polling, #107880).  Walking more remote IPs cannot help and only mints more
TIME_WAIT sockets, so the walk must stop and surface the error.

Contracts asserted here (mutation-survivable)
----------------------------------------------
- A plain connect error stays retryable (the fallback walk is preserved).
- An ``EADDRNOTAVAIL`` / Windows 10049 code anywhere in the exception chain makes
  the connect error non-retryable, and the end-to-end walk stops at the first path
  instead of trying every literal.
- Non-connect errors are never retryable, regardless of classification.
"""

from __future__ import annotations

import errno

import httpx
import pytest

import plugins.platforms.telegram.telegram_network as tnet


def _exhaustion_error(message="Cannot assign requested address") -> httpx.ConnectError:
    err = httpx.ConnectError(message)
    err.__cause__ = OSError(errno.EADDRNOTAVAIL, message)
    return err


class TestIsRetryableConnectError:
    def test_plain_connect_error_stays_retryable(self):
        assert tnet._is_retryable_connect_error(httpx.ConnectError("connection refused")) is True

    def test_connect_timeout_stays_retryable(self):
        assert tnet._is_retryable_connect_error(httpx.ConnectTimeout("timed out")) is True

    def test_eaddrnotavail_in_cause_is_not_retryable(self):
        assert tnet._is_retryable_connect_error(_exhaustion_error()) is False

    def test_wsa_eaddrnotavail_code_is_not_retryable(self):
        err = httpx.ConnectError("win error")
        err.__cause__ = OSError(10049, "win error")
        assert tnet._is_retryable_connect_error(err) is False

    def test_deep_context_chain_is_searched(self):
        root = OSError(errno.EADDRNOTAVAIL, "exhausted")
        mid = httpx.ConnectError("wrapped")
        mid.__context__ = root
        assert tnet._is_retryable_connect_error(mid) is False

    def test_unrelated_errno_stays_retryable(self):
        err = httpx.ConnectError("refused")
        err.__cause__ = OSError(errno.ECONNREFUSED, "refused")
        assert tnet._is_retryable_connect_error(err) is True

    def test_non_connect_errors_are_never_retryable(self):
        assert tnet._is_retryable_connect_error(ValueError("nope")) is False
        assert tnet._is_retryable_connect_error(_exhaustion_error().__cause__) is False


class TestFallbackWalkStopsOnLocalExhaustion:
    @pytest.mark.asyncio
    async def test_exhaustion_on_first_literal_stops_the_walk(self, monkeypatch):
        calls = []
        behavior = {"149.154.167.220": _exhaustion_error(), "149.154.167.221": "ok"}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.167.221"])

        with pytest.raises(httpx.ConnectError):
            await transport.handle_async_request(_telegram_request())

        # Only the first literal was dialed; no second IP, no hostname fallback.
        assert [c["url_host"] for c in calls] == ["149.154.167.220"]

    @pytest.mark.asyncio
    async def test_ordinary_connect_error_still_walks_the_fallbacks(self, monkeypatch):
        calls = []
        behavior = {"149.154.167.220": "connect_error", "149.154.167.221": "ok"}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.167.221"])
        resp = await transport.handle_async_request(_telegram_request())

        assert resp.status_code == 200
        assert [c["url_host"] for c in calls] == ["149.154.167.220", "149.154.167.221"]


def _fake_transport_factory(calls, behavior):
    """Returns a factory that creates FakeTransport instances."""
    instances = []

    class FakeTransport(httpx.AsyncBaseTransport):
        def __init__(self, **kwargs):
            self.calls = calls
            self.behavior = behavior

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls.append({"url_host": request.url.host})
            action = self.behavior.get(request.url.host, "ok")
            if isinstance(action, Exception):
                raise action
            if action == "connect_error":
                raise httpx.ConnectError("connect error")
            return httpx.Response(200, request=request, text="ok")

        async def aclose(self) -> None:
            pass

    def factory(**kwargs):
        t = FakeTransport(**kwargs)
        instances.append(t)
        return t

    factory.instances = instances
    return factory


def _telegram_request(path="/botTOKEN/getMe"):
    return httpx.Request("GET", f"https://api.telegram.org{path}")
