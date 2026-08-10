"""Regression test for #71593 / #63311 — Telegram fallback-pool FD leak.

Background
----------
``TelegramFallbackTransport`` (plugins/platforms/telegram/telegram_network.py)
routes Telegram Bot API requests via per-IP fallback ``httpx`` pools when the
primary DNS path is unreachable.  The pre-fix version built one
``AsyncHTTPTransport`` per fallback IP eagerly in ``__init__`` and *never* tore
them down: on a retryable connect failure the handler only logged and
continued, so a socket left in ``CLOSE_WAIT`` by a peer-closed connection stayed
inside the retained pool.  Each retry leaked another file descriptor until the
gateway hit EMFILE and wedged (accept(), config reads and DNS all failing).

The fix (this PR):
  * builds fallback pools lazily via ``_get_fallback`` (nothing in ``__init__``);
  * on a retryable connect failure calls ``_reset_fallback`` which pops the
    poisoned pool out of ``self._fallbacks`` and ``aclose()``s it, so its dead
    sockets are released instead of accumulating;
  * bounds every pool at ``Limits(max_connections=8)`` as a *default* via
    ``setdefault`` — a caller-supplied ``limits`` kwarg still wins.

Contract asserted here (mutation-survivable)
---------------------------------------------
1. A retryable connect failure on a fallback IP causes that pool to be
   ``aclose()``d and dropped from ``self._fallbacks`` — NOT retained.  This is
   the discard-on-failure path: revert the ``_reset_fallback`` call in
   ``handle_async_request`` and ``test_failed_fallback_pool_is_discarded_and_closed``
   fails (the pool is retained and never closed).
2. A caller-supplied ``limits`` kwarg wins over the ``_POOL_LIMITS`` default.
"""

import httpx
import pytest

import plugins.platforms.telegram.telegram_network as tnet


def _telegram_request(path="/botTOKEN/getMe"):
    return httpx.Request("GET", f"https://api.telegram.org{path}")


class _CountingTransport(httpx.AsyncBaseTransport):
    """Fake AsyncHTTPTransport: fails/succeeds per host, records aclose()."""

    def __init__(self, behavior, closed_log):
        self.behavior = behavior
        self.closed_log = closed_log
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        action = self.behavior.get(request.url.host, "ok")
        if action == "timeout":
            raise httpx.ConnectTimeout("timed out")
        if action == "connect_error":
            raise httpx.ConnectError("connect error")
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        self.closed = True
        self.closed_log.append(self)


def _factory(behavior, closed_log, kwargs_log=None):
    def factory(**kwargs):
        if kwargs_log is not None:
            kwargs_log.append(kwargs)
        return _CountingTransport(behavior, closed_log)

    return factory


@pytest.mark.asyncio
async def test_failed_fallback_pool_is_discarded_and_closed(monkeypatch):
    """The discard-on-failure path: a retryable connect failure on a fallback
    IP must aclose() that pool and drop it from ``self._fallbacks`` so its
    CLOSE_WAIT socket is released (#71593 / #63311).

    Sabotage check: reverting the ``await self._reset_fallback(ip)`` call in
    ``handle_async_request`` retains the poisoned pool → this test fails
    (pool still in ``_fallbacks`` and never aclose()d).
    """
    closed_log: list = []
    # Primary + both fallback IPs fail with a retryable connect error, so every
    # fallback pool that gets built must also get discarded.
    behavior = {
        "api.telegram.org": "timeout",
        "149.154.167.220": "connect_error",
        "149.154.167.221": "timeout",
    }
    monkeypatch.setattr(
        tnet.httpx, "AsyncHTTPTransport", _factory(behavior, closed_log)
    )

    transport = tnet.TelegramFallbackTransport(
        ["149.154.167.220", "149.154.167.221"]
    )

    # All paths fail → the last error propagates.
    with pytest.raises((httpx.ConnectTimeout, httpx.ConnectError)):
        await transport.handle_async_request(_telegram_request())

    # The poisoned fallback pools must NOT be retained — they were discarded.
    assert transport._fallbacks == {}, (
        "Failed fallback pools were retained in self._fallbacks — the "
        "CLOSE_WAIT sockets leak (revert of _reset_fallback? #71593)."
    )

    # The failed primary plus each fallback pool must be
    # aclose()d exactly once (one primary + two fallback IPs).
    assert len(closed_log) == 3, (
        f"Expected 3 discarded/closed transports, got {len(closed_log)} — "
        "the discard-on-failure path did not aclose() the poisoned pools."
    )
    assert all(t.closed for t in closed_log)
    await transport.aclose()


@pytest.mark.asyncio
async def test_failed_primary_pool_is_discarded_and_closed(monkeypatch):
    """A failed primary attempt must release its pool before fallback (#82920)."""
    for key in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy",
        "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    behavior = {"api.telegram.org": "timeout", "149.154.167.220": "ok"}
    instances = []

    def factory(**kwargs):
        transport = _CountingTransport(behavior, [])
        instances.append(transport)
        return transport

    monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)
    transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
    try:
        response = await transport.handle_async_request(_telegram_request())
        assert response.status_code == 200
        assert len(instances) == 3
        assert instances[0].closed
        assert not instances[1].closed
        assert not instances[2].closed
    finally:
        await transport.aclose()


def test_caller_limits_win_over_pool_default(monkeypatch):
    """A caller-supplied ``limits`` kwarg must win over the ``_POOL_LIMITS``
    ``setdefault`` default, for both the primary and lazily-built fallback
    pools (#71593)."""
    import asyncio

    kwargs_log: list = []
    for key in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy",
        "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        tnet.httpx, "AsyncHTTPTransport", _factory({}, [], kwargs_log)
    )

    custom_limits = httpx.Limits(
        max_connections=42, max_keepalive_connections=10, keepalive_expiry=30.0
    )
    transport = tnet.TelegramFallbackTransport(
        ["149.154.167.220"], limits=custom_limits
    )
    # Primary built in __init__ with the caller's limits (not the default).
    assert kwargs_log[0]["limits"] is custom_limits

    # Lazily-built fallback pool must also carry the caller's limits.
    asyncio.run(transport._get_fallback("149.154.167.220"))
    assert len(kwargs_log) == 2
    assert all(kw["limits"] is custom_limits for kw in kwargs_log)
    # And the caller's limits are NOT the class default.
    assert custom_limits is not tnet.TelegramFallbackTransport._POOL_LIMITS


def test_pool_default_limits_applied_when_caller_omits(monkeypatch):
    """When the caller supplies no ``limits``, the bounded ``_POOL_LIMITS``
    default (max_connections=8) is applied via setdefault (#71593)."""
    kwargs_log: list = []
    for key in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy",
        "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        tnet.httpx, "AsyncHTTPTransport", _factory({}, [], kwargs_log)
    )

    transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
    limits = kwargs_log[0]["limits"]
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 8
    assert limits is transport._POOL_LIMITS
