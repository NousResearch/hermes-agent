"""The fallback path walk shares one CONNECT budget per request (#136412).

The adapter bounds the whole Telegram send with a wall-clock deadline, but the httpx client's
connect timeout applies PER fallback path (sticky → IPv4 literals → hostname). Several hung paths
sum past the send deadline; it then fires mid-walk and surfaces as a bare ``TimeoutError`` with no
cause, which the send classifier must treat as non-retryable (a read timeout may have delivered),
so the delivery is lost to ``abandoned``. ``connect_budget`` bounds the walk: the sticky path is
front-loaded (it gets everything but one average share held back, so at stock defaults it keeps the
client's full connect timeout), later paths split the remainder, a spent budget surfaces the
underlying ``ConnectTimeout`` immediately — which IS classified safely retryable — and tighter
configured connects are never widened.
"""

import asyncio

import httpx
import pytest

import plugins.platforms.telegram.telegram_network as tnet


class RecordingTransport(httpx.AsyncBaseTransport):
    """Records per-request ``extensions["timeout"]`` and raises/returns per host→action. When a
    ``clock`` is given, each attempt advances it by ``burn`` seconds (the wall-clock cost of a hung
    connect) so budget-exhaustion scenarios are deterministic."""

    def __init__(self, calls, behavior, clock=None, burn=0.0):
        self.calls = calls
        self.behavior = behavior
        self.clock = clock
        self.burn = burn

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.clock is not None:
            self.clock.now += self.burn
        self.calls.append(
            {"url_host": request.url.host, "timeout": request.extensions.get("timeout")}
        )
        action = self.behavior.get(request.url.host, "ok")
        if isinstance(action, Exception):
            raise action
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        return None


def _recording_factory(calls, behavior, clock=None, burn=0.0):
    def factory(**kwargs):
        return RecordingTransport(calls, behavior, clock, burn)

    return factory


class _FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def monotonic(self) -> float:
        return self.now


def _budget_request(connect=10.0, read=20.0, write=20.0, pool=8.0):
    request = httpx.Request("GET", "https://api.telegram.org/botTOKEN/getMe")
    request.extensions["timeout"] = {
        "connect": connect,
        "read": read,
        "write": write,
        "pool": pool,
    }
    return request


_IPS = ["149.154.166.110", "149.154.167.220"]
_ALL_TIMEOUT = {
    "149.154.166.110": httpx.ConnectTimeout("ip1 connect timeout"),
    "149.154.167.220": httpx.ConnectTimeout("ip2 connect timeout"),
    "api.telegram.org": httpx.ConnectTimeout("hostname connect timeout"),
}


class TestConnectBudgetWalk:
    @pytest.mark.asyncio
    async def test_sticky_path_is_front_loaded_within_the_budget(self, monkeypatch):
        # Frozen clock: the shares are exact. The sticky first path gets the budget minus one average
        # share held back for failover (15 − 15/3 = 10.0 — at stock defaults that equals the client's
        # full connect timeout, so the budget costs it nothing); later paths split the remainder
        # (15/2 = 7.5), and the walk can never outspend the budget.
        calls = []
        clock = _FakeClock(100.0)
        monkeypatch.setattr(tnet.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(
            tnet.httpx, "AsyncHTTPTransport", _recording_factory(calls, _ALL_TIMEOUT)
        )

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=15.0)
        with pytest.raises(httpx.ConnectTimeout):
            await transport.handle_async_request(_budget_request(connect=20.0))

        # Order: ip1 → ip2 → hostname. Shares 15 − 15/3, 15/2, 15/1 = 10.0 / 7.5 / 15.0 — the sticky
        # path's slice is the front-loaded one; the frozen clock keeps `remaining` flat, so the last
        # path's share is the whole budget (a live clock's earlier burns would have spent it).
        assert [c["url_host"] for c in calls] == [
            "149.154.166.110",
            "149.154.167.220",
            "api.telegram.org",
        ]
        assert [c["timeout"]["connect"] for c in calls] == [10.0, 7.5, 15.0]
        # The other phases pass through untouched on every path.
        for call in calls:
            assert call["timeout"]["read"] == 20.0
            assert call["timeout"]["pool"] == 8.0

    @pytest.mark.asyncio
    async def test_stock_client_connect_survives_the_budget_untouched(
        self, monkeypatch
    ):
        # Front-loading means stock defaults (10s client connect, 15s budget, 3 paths) leave the
        # sticky path's connect UNCHANGED: its share (15 − 5 = 10.0) never narrows the configured
        # 10s, so a slow-but-live sticky path keeps exactly the window it had before #136412.
        calls = []
        clock = _FakeClock(100.0)
        monkeypatch.setattr(tnet.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(
            tnet.httpx, "AsyncHTTPTransport", _recording_factory(calls, _ALL_TIMEOUT)
        )

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=15.0)
        with pytest.raises(httpx.ConnectTimeout):
            await transport.handle_async_request(_budget_request())

        # ip1's share is exactly the configured 10.0 (not narrowed); ip2 gets 15/2 = 7.5; the
        # hostname path's 15.0 share exceeds the configured 10.0, which wins (never widened).
        assert [c["timeout"]["connect"] for c in calls] == [10.0, 7.5, 10.0]

    @pytest.mark.asyncio
    async def test_spent_budget_surfaces_last_connect_error_instead_of_walking_on(
        self, monkeypatch
    ):
        # Each path attempt burns 6s of wall clock against a 10s budget: path 1 (share 10 − 10/3)
        # and path 2 (share 4/2) run and fail; path 3 would start at t=12 with nothing left, so the
        # walk stops and raises path 2's ConnectTimeout instead of arming an attempt the caller's
        # send deadline would interrupt and mask.
        calls = []
        clock = _FakeClock(0.0)
        monkeypatch.setattr(tnet.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(
            tnet.httpx,
            "AsyncHTTPTransport",
            _recording_factory(calls, _ALL_TIMEOUT, clock=clock, burn=6.0),
        )

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=10.0)
        with pytest.raises(httpx.ConnectTimeout) as excinfo:
            await transport.handle_async_request(_budget_request())

        assert len(calls) == 2
        assert excinfo.value is _ALL_TIMEOUT["149.154.167.220"]

    @pytest.mark.asyncio
    async def test_tighter_configured_connect_timeout_is_not_widened(self, monkeypatch):
        # The budget only ever NARROWS the connect phase: a client already configured tighter (here
        # 1.0s against a 10.0s front-loaded first share) keeps its own bound on every path.
        calls = []
        behavior = {"149.154.166.110": httpx.ConnectTimeout("ip1 connect timeout")}
        clock = _FakeClock(100.0)
        monkeypatch.setattr(tnet.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(
            tnet.httpx, "AsyncHTTPTransport", _recording_factory(calls, behavior)
        )

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=15.0)
        response = await transport.handle_async_request(_budget_request(connect=1.0))

        assert response.status_code == 200
        assert calls[0]["url_host"] == "149.154.166.110"
        assert calls[0]["timeout"]["connect"] == 1.0
        assert calls[1]["timeout"]["connect"] == 1.0

    @pytest.mark.asyncio
    async def test_no_budget_keeps_legacy_per_request_timeouts(self, monkeypatch):
        # connect_budget=None (default): the walk leaves the client's per-request timeout dict alone.
        calls = []
        behavior = {"149.154.166.110": httpx.ConnectTimeout("ip1 connect timeout")}
        monkeypatch.setattr(
            tnet.httpx, "AsyncHTTPTransport", _recording_factory(calls, behavior)
        )

        transport = tnet.TelegramFallbackTransport(_IPS)
        response = await transport.handle_async_request(_budget_request(connect=10.0))

        assert response.status_code == 200
        assert all(c["timeout"]["connect"] == 10.0 for c in calls)

    @pytest.mark.asyncio
    async def test_request_without_timeout_dict_is_untouched(self, monkeypatch):
        # httpx clients always stamp extensions["timeout"], but a bare request (direct transport
        # call, tests) must not get one injected: a fresh dict would drop the other phases'
        # transport defaults.
        calls = []
        behavior = {"149.154.166.110": httpx.ConnectTimeout("ip1 connect timeout")}
        monkeypatch.setattr(tnet.time, "monotonic", lambda: 100.0)
        monkeypatch.setattr(
            tnet.httpx, "AsyncHTTPTransport", _recording_factory(calls, behavior)
        )

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=15.0)
        response = await transport.handle_async_request(
            httpx.Request("GET", "https://api.telegram.org/botTOKEN/getMe")
        )

        assert response.status_code == 200
        assert all(not c["timeout"] for c in calls)


class _HangingTransport(httpx.AsyncBaseTransport):
    """Hangs every path for its assigned CONNECT share (a real ``asyncio.sleep``, so the walk's
    ``time.monotonic`` and the deadline's thread timer see the same clock), then raises
    ``ConnectTimeout`` — the shape of a blackholed path."""

    def __init__(self):
        self.connects: list[float | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        connect = request.extensions.get("timeout", {}).get("connect")
        self.connects.append(connect)
        await asyncio.sleep(connect)
        raise httpx.ConnectTimeout(f"connect timeout after {connect}s")

    async def aclose(self) -> None:
        return None


class TestBudgetedWalkClassifiesRetryable:
    """End-to-end shape of the #136412 repro: the adapter's wall-clock send deadline wrapped
    around a 3-path all-hang walk. Before the budget, the walk outlived the deadline and the
    caller got a bare ``TimeoutError`` — classified NON-retryable (a read timeout may have
    delivered), so the delivery was abandoned. With ``connect_budget`` the walk finishes inside
    the deadline and the underlying ``ConnectTimeout`` surfaces, which the send classifier reads
    as safely retryable. Durations are scaled down (30s/15s → 2s/1s) but both clocks are real, so
    the test drives the actual ``_await_with_thread_deadline`` + ``time.monotonic`` machinery."""

    @pytest.mark.asyncio
    async def test_all_hang_walk_beats_the_deadline_and_classifies_retryable(
        self, monkeypatch
    ):
        from plugins.platforms.telegram.adapter import (
            TelegramAdapter,
            _await_with_thread_deadline,
        )

        hanging = _HangingTransport()
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", lambda **kwargs: hanging)

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=1.0)
        with pytest.raises(httpx.ConnectTimeout) as excinfo:
            await _await_with_thread_deadline(
                transport.handle_async_request(_budget_request(connect=10.0)),
                timeout=2.0,
                label="telegram-send",
                dump_on_blocked_loop=False,
            )

        # All three paths were attempted and their CONNECT shares stayed inside the 1.0s budget
        # (front-loaded 1 − 1/3, then the split of what was left), so the walk finished well
        # inside the 2.0s deadline: the escapee is the underlying ConnectTimeout, NOT the bare
        # deadline TimeoutError the repro showed.
        assert len(hanging.connects) == 3
        assert all(c is not None and c < 1.0 for c in hanging.connects)
        assert hanging.connects[0] > hanging.connects[1]
        # The retry contract: a ConnectTimeout means TCP never connected, so re-sending is safe —
        # the classifier the send ladder uses must read it as retryable (the bare deadline
        # TimeoutError is exactly the case it must keep refusing).
        assert TelegramAdapter._looks_like_connect_timeout(excinfo.value) is True
        assert TelegramAdapter._is_timed_out(excinfo.value) is False
        assert (
            TelegramAdapter._looks_like_connect_timeout(
                TimeoutError("timed out after 2s (telegram-send)")
            )
            is False
        )
