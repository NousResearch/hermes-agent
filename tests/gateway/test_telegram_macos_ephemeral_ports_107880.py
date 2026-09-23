"""Regression tests for #107880 — Darwin getUpdates keepalive + local EADDRNOTAVAIL.

On macOS, ``max_keepalive_connections=0`` for the getUpdates pool forces a
fresh TCP/TLS 4-tuple per long-poll. After ~2 days / two Telegram profiles,
TIME_WAIT fills the 16384-port ephemeral range and unrelated HTTPS fails
with ``EADDRNOTAVAIL``.

Darwin must reuse getUpdates sockets (keepalive >= 1). Windows stays at 0
(#87057). Linux stays at 0 (fail-open). Local ephemeral-port exhaustion is
not a remote-IP failure: ``_is_retryable_connect_error`` must not walk
fallback IPs on ``EADDRNOTAVAIL`` / ``WSAEADDRNOTAVAIL``.

Platform decisions are tested via platform-as-data on the pure helper
(``platform=...``), not by monkeypatching ``sys.platform``.
"""

from __future__ import annotations

import asyncio
import errno
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
import plugins.platforms.telegram.telegram_network as tnet


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

    async def _no_fallback():
        return list(fallback_ips or [])

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback)
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *a, **k: proxy_url)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _RecordingHTTPXRequest)

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    if fallback_ips is not None:
        monkeypatch.setattr(adapter, "_fallback_ips", lambda: list(fallback_ips))

    chainable = MagicMock()
    chainable.token.return_value = chainable
    chainable.base_url.return_value = chainable
    chainable.base_file_url.return_value = chainable
    chainable.local_mode.return_value = chainable
    chainable.request.return_value = chainable
    chainable.get_updates_request.return_value = chainable
    chainable.build.side_effect = _StopConnect

    builder_root = MagicMock()
    builder_root.builder.return_value = chainable
    monkeypatch.setattr(tg_adapter, "Application", builder_root)

    try:
        asyncio.run(adapter.connect())
    except _StopConnect:
        pass
    except Exception:
        pass

    return list(_RecordingHTTPXRequest.instances)


def _updates_limits_from_proxy(instances):
    assert len(instances) >= 2, "connect() did not build general + getUpdates HTTPXRequest"
    limits = instances[1].kwargs.get("httpx_kwargs", {}).get("limits")
    assert isinstance(limits, httpx.Limits), (
        "getUpdates HTTPXRequest must receive httpx_kwargs['limits']"
    )
    return limits


def _updates_limits_from_fallback(instances):
    assert len(instances) >= 2
    transport = instances[1].kwargs["httpx_kwargs"]["transport"]
    assert isinstance(transport, tg_adapter.TelegramFallbackTransport)
    limits = transport._transport_kwargs["limits"]
    assert isinstance(limits, httpx.Limits)
    return limits, transport


def _wrap_connect_error(inner: BaseException) -> httpx.ConnectError:
    err = httpx.ConnectError("All connection attempts failed")
    err.__cause__ = inner
    return err


def test_getupdates_keepalive_platform_as_data():
    """#107880: platform-as-data — Darwin reuses; win32/linux stay at 0."""
    base = SimpleNamespace(max_keepalive_connections=10)
    assert tg_adapter._getupdates_max_keepalive_connections(base, platform="darwin") == 10
    assert tg_adapter._getupdates_max_keepalive_connections(base, platform="win32") == 0
    assert tg_adapter._getupdates_max_keepalive_connections(base, platform="linux") == 0

    floor = SimpleNamespace(max_keepalive_connections=0)
    assert tg_adapter._getupdates_max_keepalive_connections(floor, platform="darwin") == 1
    missing = SimpleNamespace()
    assert tg_adapter._getupdates_max_keepalive_connections(missing, platform="darwin") == 1


def test_proxy_branch_updates_pool_uses_helper_keepalive(monkeypatch):
    """connect() proxy branch must wire the helper's keepalive into getUpdates limits."""
    monkeypatch.setattr(
        tg_adapter, "_getupdates_max_keepalive_connections", lambda base_limits, platform=None: 7
    )
    instances = _drive_connect(monkeypatch, proxy_url="http://127.0.0.1:9/")
    limits = _updates_limits_from_proxy(instances)
    assert limits.max_keepalive_connections == 7


def test_fallback_branch_updates_transport_uses_helper_keepalive(monkeypatch):
    """connect() fallback branch must wire the helper's keepalive into the inner transport."""
    monkeypatch.setattr(
        tg_adapter, "_getupdates_max_keepalive_connections", lambda base_limits, platform=None: 7
    )
    monkeypatch.delenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", raising=False)
    instances = _drive_connect(
        monkeypatch, proxy_url=None, fallback_ips=["149.154.167.220"]
    )
    limits, transport = _updates_limits_from_fallback(instances)
    assert limits.max_keepalive_connections == 7
    asyncio.run(transport.aclose())
    asyncio.run(instances[0].kwargs["httpx_kwargs"]["transport"].aclose())


def test_eaddrnotavail_connect_error_is_not_retryable():
    """Local ephemeral-port exhaustion is not a remote-IP failure (#107880)."""
    inner = OSError(errno.EADDRNOTAVAIL, "Can't assign requested address")
    wrapped = _wrap_connect_error(inner)
    assert tnet._is_retryable_connect_error(wrapped) is False

    win = OSError(10049, "Cannot assign requested address")
    assert tnet._is_retryable_connect_error(_wrap_connect_error(win)) is False

    via_context = httpx.ConnectError("connect error")
    via_context.__context__ = OSError(errno.EADDRNOTAVAIL, "Can't assign requested address")
    assert tnet._is_retryable_connect_error(via_context) is False


def test_generic_connect_error_is_still_retryable():
    """Fail-open: generic ConnectError and remote-connect OSErrors stay retryable."""
    assert tnet._is_retryable_connect_error(httpx.ConnectError("connect error")) is True
    assert tnet._is_retryable_connect_error(httpx.ConnectTimeout("timed out")) is True
    refused = _wrap_connect_error(OSError(errno.ECONNREFUSED, "Connection refused"))
    assert tnet._is_retryable_connect_error(refused) is True
    timed_out = _wrap_connect_error(OSError(errno.ETIMEDOUT, "timed out"))
    assert tnet._is_retryable_connect_error(timed_out) is True
    unreachable = _wrap_connect_error(OSError(errno.EHOSTUNREACH, "No route to host"))
    assert tnet._is_retryable_connect_error(unreachable) is True


def test_stale_reuse_read_errors_do_not_walk_fallback_ips():
    """Stale pooled-socket failures must not enter the fallback-IP walk.

    ``_is_retryable_connect_error`` only classifies ConnectTimeout/ConnectError.
    A server-FIN'd idle connection surfaces as ReadError/RemoteProtocolError on
    the next getUpdates read — those must raise immediately (no IP churn).
    """
    assert tnet._is_retryable_connect_error(httpx.ReadError("Connection reset")) is False
    assert tnet._is_retryable_connect_error(
        httpx.RemoteProtocolError("Server disconnected without sending a response.")
    ) is False
