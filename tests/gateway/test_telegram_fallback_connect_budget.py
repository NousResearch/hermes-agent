"""One request's fallback path walk shares a connect budget that ends inside the send deadline.

The adapter bounds every text send with a 30s wall-clock deadline, but httpx's connect timeout
applies per path. Three hung paths at the stock 10s connect timeout used up the whole deadline. The
deadline then fired mid-walk as a bare TimeoutError, which the send classifier treats as
"may have been delivered", so the message was dropped with no retry.
"""

import httpx
import pytest

import hermes_agent.plugins.platforms.telegram.telegram_network as tnet


class _Clock:
    now = 1000.0

    def monotonic(self):
        return self.now


class _HangingConnect(httpx.AsyncBaseTransport):
    """An inner pool whose connect hangs for the request's whole connect timeout (advancing the
    fake clock by it), then times out."""

    def __init__(self, calls, clock):
        self.calls, self.clock = calls, clock

    async def handle_async_request(self, request):
        connect = request.extensions["timeout"]["connect"]
        self.calls.append((request.url.host, connect))
        self.clock.now += connect
        raise httpx.ConnectTimeout("connect timed out", request=request)

    async def aclose(self):
        return None


class _CaptureRequest:
    """Stands in for PTB's HTTPXRequest: keeps the httpx kwargs build_ptb_requests hands it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _stock_send_request():
    request = httpx.Request("POST", "https://api.telegram.org/botTOKEN/sendMessage")
    timeouts = tnet.ptb_request_kwargs()
    request.extensions["timeout"] = {"connect": timeouts["connect_timeout"], "read": timeouts["read_timeout"],
                                     "write": timeouts["write_timeout"], "pool": timeouts["pool_timeout"]}
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize("which", [0, 1], ids=["general", "getUpdates"])
async def test_built_transports_walk_every_path_within_the_send_deadline(monkeypatch, which):
    """Every path (two seed IPs plus the hostname) is tried, and the walk ends with the underlying
    ConnectTimeout. It must not run into the send deadline."""
    clock = _Clock()
    monkeypatch.setattr(tnet.time, "monotonic", clock.monotonic)
    calls = []
    monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", lambda **kw: _HangingConnect(calls, clock))
    requests = tnet.build_ptb_requests(_CaptureRequest, fallback_ips=list(tnet.SEED_FALLBACK_IPS), proxy_url=None,
                                       verify=False)
    transport = requests[which].kwargs["httpx_kwargs"]["transport"]

    with pytest.raises(httpx.ConnectTimeout):
        await transport.handle_async_request(_stock_send_request())

    assert [host for host, _ in calls] == [*tnet.SEED_FALLBACK_IPS, "api.telegram.org"]
    # The sticky/first path keeps the stock 10s connect; the walk total stays well inside 30s.
    assert calls[0][1] == 10.0
    assert sum(connect for _, connect in calls) == 15.0


@pytest.mark.asyncio
async def test_tighter_configured_connect_is_never_widened(monkeypatch):
    calls = []
    monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", lambda **kw: _HangingConnect(calls, _Clock()))
    transport = tnet.TelegramFallbackTransport(tnet.SEED_FALLBACK_IPS, connect_budget=15.0)
    request = _stock_send_request()
    request.extensions["timeout"]["connect"] = 1.0

    with pytest.raises(httpx.ConnectTimeout):
        await transport.handle_async_request(request)

    assert [connect for _, connect in calls] == [1.0, 1.0, 1.0]
