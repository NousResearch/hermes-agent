"""Regression test for #31599 — Telegram general-pool CLOSE_WAIT fd leak.

Background
----------
PTB's ``telegram.request.HTTPXRequest`` builds the underlying
``httpx.AsyncClient`` with ``limits = httpx.Limits(max_connections=...)``
and *no* keepalive tuning, so httpx's default ``keepalive_expiry=5.0``
applies.  Behind an HTTP proxy (Cloudflare Warp etc.) a peer-initiated
FIN can sit in ``CLOSE_WAIT`` longer than that, leaking fds in the
general request pool (``_request[1]`` — the pool that routes
``bot.send_message`` / ``set_my_commands``), which
``_drain_polling_connections`` never resets.

The fix wires the shared ``gateway.platforms._http_client_limits``
``platform_httpx_limits()`` helper into *every* HTTPXRequest the adapter
builds — the fallback-transport branch, the proxy branch, and the plain
branch — so idle keepalive sockets drain aggressively.

Contracts asserted here (mutation-survivable)
----------------------------------------------
Proxy and direct-DNS ``HTTPXRequest`` instances must receive
``httpx_kwargs["limits"]`` with a ``keepalive_expiry`` strictly below
httpx's 5.0 default.  The fallback-IP instances must pass equivalent
limits into both inner ``AsyncHTTPTransport`` pools because httpx ignores
client-level limits when a custom transport is supplied.
"""

import asyncio
import socket
from unittest.mock import MagicMock

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class _StopConnect(Exception):
    """Sentinel raised to abort connect() once requests are built."""


class _RecordingHTTPXRequest:
    """Stand-in for PTB's HTTPXRequest that records constructor kwargs."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _RecordingHTTPXRequest.instances.append(self)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _drive_connect(monkeypatch, *, proxy_url, fallback_ips=None):
    """Run connect() far enough to build the HTTPXRequests, then abort.

    Returns the list of recorded _RecordingHTTPXRequest instances.
    """
    _RecordingHTTPXRequest.instances = []

    # No DoH auto-discovery → exercise the proxy / plain branches, not fallback.
    async def _no_fallback():
        return list(fallback_ips or [])

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback)
    monkeypatch.setattr(
        tg_adapter, "resolve_proxy_url", lambda *a, **k: proxy_url
    )
    # Replace the real HTTPXRequest with our recorder.
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _RecordingHTTPXRequest)

    adapter = _make_adapter()
    # Skip the cross-process token lock.
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    # Ensure the adapter reports no statically-configured fallback IPs.
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    # Proxy/direct tests must not silently fall into seed fallback-IP discovery;
    # the explicit fallback test below opts back in with supplied addresses.
    if fallback_ips is None:
        monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")

    if fallback_ips is not None:
        monkeypatch.delenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", raising=False)
        monkeypatch.setattr(adapter, "_fallback_ips", lambda: list(fallback_ips))

    # Build only the requests under test.  Going through connect() retries and
    # rebuilds requests after its sentinel exception, which obscures the branch
    # selected for this test.
    asyncio.run(adapter._build_ptb_requests())
    return list(_RecordingHTTPXRequest.instances)


def _assert_keepalive_tight(instances):
    assert instances, "connect() built no HTTPXRequest — test setup is wrong"
    for inst in instances:
        limits = inst.kwargs.get("httpx_kwargs", {}).get("limits")
        assert isinstance(limits, httpx.Limits), (
            "HTTPXRequest must receive httpx_kwargs['limits'] = httpx.Limits "
            "wired from platform_httpx_limits() (#31599). Missing → PTB falls "
            "back to default keepalive_expiry=5.0 and leaks CLOSE_WAIT fds."
        )
        # The whole point: keepalive must be tighter than httpx's 5.0 default.
        assert limits.keepalive_expiry is not None
        assert limits.keepalive_expiry < 5.0, (
            "keepalive_expiry must be < httpx default 5.0 so idle/CLOSE_WAIT "
            "sockets drain promptly behind a proxy (#31599)."
        )
        assert limits.max_keepalive_connections is not None
        assert 1 <= limits.max_keepalive_connections <= 50
        # PTB's connection_pool_size (max_connections) must be preserved.
        assert limits.max_connections is not None and limits.max_connections > 0


def _assert_updates_pool_never_reuses(instance):
    """The long-poll pool must not reuse server-closed connections (#87057)."""
    limits = instance.kwargs.get("httpx_kwargs", {}).get("limits")
    assert isinstance(limits, httpx.Limits)
    assert limits.max_keepalive_connections == 0
    assert limits.max_connections == 512


def test_proxy_branch_general_pool_has_tight_keepalive(monkeypatch):
    """The proxy path the #31599 reporter hit must wire tuned limits."""
    monkeypatch.delenv("HERMES_TELEGRAM_PROXY_OVERRIDE", raising=False)
    instances = _drive_connect(monkeypatch, proxy_url="http://127.0.0.1:9/")
    # Both the general request pool and the get_updates pool are built here.
    assert len(instances) >= 2
    _assert_keepalive_tight(instances[:1])
    _assert_updates_pool_never_reuses(instances[1])
    # Sanity: the proxy was actually threaded through (we're on the proxy branch).
    assert any(inst.kwargs.get("proxy") == "http://127.0.0.1:9/" for inst in instances)
    # An explicit Telegram proxy must not also inherit the gateway's generic
    # HTTP(S)_PROXY route through httpx's environment handling.
    assert all(inst.kwargs["httpx_kwargs"].get("trust_env") is False for inst in instances[:2])


def test_fallback_branch_forwards_tuned_limits_to_inner_transports(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)

    instances = _drive_connect(
        monkeypatch,
        proxy_url=None,
        fallback_ips=["149.154.167.220"],
    )

    assert len(instances) >= 2
    for index, instance in enumerate(instances):
        transport = instance.kwargs["httpx_kwargs"]["transport"]
        assert isinstance(transport, tg_adapter.TelegramFallbackTransport)
        limits = transport._transport_kwargs["limits"]
        assert isinstance(limits, httpx.Limits)
        assert limits.keepalive_expiry is not None
        assert limits.keepalive_expiry < 5.0
        assert limits.max_connections == 512
        sock_opts = transport._transport_kwargs.get("socket_options")
        assert sock_opts, "fallback transport must enable TCP keepalive (#87057)"
        assert any(
            opt[0] == socket.SOL_SOCKET
            and opt[1] == socket.SO_KEEPALIVE
            and opt[2] == 1
            for opt in sock_opts
        )
        if index == 0:
            assert limits.max_keepalive_connections >= 1
        else:
            assert limits.max_keepalive_connections == 0

    for instance in instances:
        asyncio.run(instance.kwargs["httpx_kwargs"]["transport"].aclose())
