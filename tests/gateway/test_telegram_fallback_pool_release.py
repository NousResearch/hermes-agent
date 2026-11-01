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

import asyncio

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
        # IPv4-first (#87015): .220 succeeds on the first try, so the
        # dual-stack hostname pool is never opened and never discarded.
        # Instances: 1 primary (unused this request) + 1 fallback (.220).
        assert len(instances) == 2
        assert not instances[0].closed
        assert not instances[1].closed
        assert transport._sticky_ip == "149.154.167.220"
    finally:
        await transport.aclose()




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
    assert limits is transport._POOL_LIMITS


class _SharedPool(httpx.AsyncBaseTransport):
    """Fake pool shared by concurrent sends: ``/fail`` raises ConnectError, other requests wait on
    ``gate``; a request still waiting when the pool is closed fails like a reset socket."""

    def __init__(self, gate):
        self.gate = gate
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fail":
            raise httpx.ConnectError("")
        await self.gate.wait()
        if self.closed:
            raise httpx.ReadError("")
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        self.closed = True


def _shared_pool_transport(monkeypatch):
    for key in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy",
        "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    gate = asyncio.Event()
    monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", lambda **kwargs: _SharedPool(gate))
    return tnet.TelegramFallbackTransport(["149.154.167.220"]), gate


@pytest.mark.asyncio
async def test_failed_path_does_not_close_pool_under_concurrent_sends(monkeypatch):
    """One ConnectError on the sticky IPv4 path used to aclose() the pool every
    concurrent send was using, turning each of them into an empty ReadError.
    The discarded pool now closes once its last request has finished (#63311)."""
    transport, gate = _shared_pool_transport(monkeypatch)
    async with httpx.AsyncClient(transport=transport) as client:
        slow = [asyncio.create_task(client.get("https://api.telegram.org/botT/sendMessage")) for _ in range(3)]
        await asyncio.sleep(0)
        shared_pool = transport._fallbacks["149.154.167.220"]

        with pytest.raises(httpx.ConnectError):
            await client.get("https://api.telegram.org/fail")

        assert not shared_pool.closed, "pool closed under in-flight sends"
        assert "149.154.167.220" not in transport._fallbacks, "failed pool must leave rotation at once"
        gate.set()
        assert [r.status_code for r in await asyncio.gather(*slow)] == [200, 200, 200]
        assert shared_pool.closed, "discarded pool must close after its last request (#63311)"


@pytest.mark.asyncio
async def test_unread_response_body_keeps_discarded_pool_open(monkeypatch):
    """A streamed body still reads from its pool: close the pool only after the body is closed."""
    transport, gate = _shared_pool_transport(monkeypatch)
    gate.set()
    shared_pool = await transport._get_fallback("149.154.167.220")
    shared_pool.handle_async_request = lambda request: _streamed_ok(request)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://api.telegram.org/botT/getFile") as response:
            await transport._reset_fallback("149.154.167.220", shared_pool)
            assert not shared_pool.closed
            assert await response.aread() == b"chunk"
        assert shared_pool.closed


async def _streamed_ok(request):
    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"chunk"

    return httpx.Response(200, request=request, stream=_Body())


@pytest.mark.asyncio
async def test_cancelled_send_releases_discarded_pool(monkeypatch):
    """A send cancelled mid-request (PTB timeouts) must not keep a discarded pool open forever."""
    transport, _gate = _shared_pool_transport(monkeypatch)
    pending = asyncio.create_task(transport.handle_async_request(_telegram_request()))
    await asyncio.sleep(0)
    shared_pool = transport._fallbacks["149.154.167.220"]
    with pytest.raises(httpx.ConnectError):
        await transport.handle_async_request(_telegram_request("/fail"))
    assert not shared_pool.closed

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert shared_pool.closed
    await transport.aclose()


@pytest.mark.asyncio
async def test_stale_failure_does_not_discard_replacement_pool(monkeypatch):
    """A failure reported against an already-discarded pool must leave its replacement alone."""
    transport, _gate = _shared_pool_transport(monkeypatch)
    stale = await transport._get_fallback("149.154.167.220")
    await transport._reset_fallback("149.154.167.220", stale)
    fresh = await transport._get_fallback("149.154.167.220")

    await transport._reset_fallback("149.154.167.220", stale)

    assert transport._fallbacks["149.154.167.220"] is fresh
    assert not fresh.closed
    await transport.aclose()
