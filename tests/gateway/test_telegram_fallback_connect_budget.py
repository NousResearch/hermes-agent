"""The fallback path walk shares one CONNECT budget per request (#136412).

The adapter bounds the whole Telegram send with a wall-clock deadline, but the httpx client's
connect timeout applies PER fallback path (sticky → IPv4 literals → hostname). Several hung paths
sum past the send deadline; it then fires mid-walk and surfaces as a bare ``TimeoutError`` with no
cause, which the send classifier must treat as non-retryable (a read timeout may have delivered),
so the delivery is lost to ``abandoned``. ``connect_budget`` bounds the walk: every path gets a
share of the remaining budget (earlier paths the largest), a spent budget surfaces the underlying
``ConnectTimeout`` immediately — which IS classified safely retryable — and tighter configured
connects are never widened.
"""

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
    async def test_each_path_gets_share_of_remaining_budget(self, monkeypatch):
        # Frozen clock: the shares are exact — paths_left divides the unspent budget, so earlier
        # (stickier) paths get the largest share and the walk can never outspend the budget.
        calls = []
        clock = _FakeClock(100.0)
        monkeypatch.setattr(tnet.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(
            tnet.httpx, "AsyncHTTPTransport", _recording_factory(calls, _ALL_TIMEOUT)
        )

        transport = tnet.TelegramFallbackTransport(_IPS, connect_budget=15.0)
        with pytest.raises(httpx.ConnectTimeout):
            await transport.handle_async_request(_budget_request())

        # Order: ip1 → ip2 → hostname. Shares 15/3, 15/2, 15/1 = 5.0 / 7.5 / 15.0; the hostname
        # path keeps the client's tighter 10.0s (a share only ever NARROWS the connect phase — with
        # a live clock the earlier paths' burns would have left it 2.5s instead).
        assert [c["url_host"] for c in calls] == [
            "149.154.166.110",
            "149.154.167.220",
            "api.telegram.org",
        ]
        assert [c["timeout"]["connect"] for c in calls] == [5.0, 7.5, 10.0]
        # The other phases pass through untouched on every path.
        for call in calls:
            assert call["timeout"]["read"] == 20.0
            assert call["timeout"]["pool"] == 8.0

    @pytest.mark.asyncio
    async def test_spent_budget_surfaces_last_connect_error_instead_of_walking_on(
        self, monkeypatch
    ):
        # Each path attempt burns 6s of wall clock against a 10s budget: path 1 (share 10/3) and
        # path 2 (share 4/2) run and fail; path 3 would start at t=12 with nothing left, so the
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
        # 1.0s against a 5.0s first share) keeps its own bound on every path.
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
