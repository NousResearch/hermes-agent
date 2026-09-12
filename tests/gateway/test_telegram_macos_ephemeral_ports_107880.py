"""Regression tests for #107880 — Darwin getUpdates keepalive + local EADDRNOTAVAIL.

On macOS, ``max_keepalive_connections=0`` for the getUpdates pool forces a
fresh TCP/TLS 4-tuple per long-poll. After ~2 days / two Telegram profiles,
TIME_WAIT fills the 16384-port ephemeral range and unrelated HTTPS fails
with ``EADDRNOTAVAIL``.

Darwin must reuse getUpdates sockets (keepalive >= 1). Windows stays at 0
(#87057). Linux stays at 0 (fail-open). Local ephemeral-port exhaustion is
not a remote-IP failure: ``_is_retryable_connect_error`` must not walk
fallback IPs on ``EADDRNOTAVAIL`` / ``WSAEADDRNOTAVAIL``.
"""

from __future__ import annotations

import asyncio
import errno
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


def test_darwin_proxy_branch_updates_pool_reuses_keepalive(monkeypatch):
    """#107880: Darwin getUpdates pool must keep >= 1 idle socket (proxy branch)."""
    monkeypatch.setattr(tg_adapter.sys, "platform", "darwin")
    instances = _drive_connect(monkeypatch, proxy_url="http://127.0.0.1:9/")
    limits = _updates_limits_from_proxy(instances)
    assert limits.max_keepalive_connections is not None
    assert limits.max_keepalive_connections >= 1, (
        "Darwin getUpdates pool must reuse sockets; max_keepalive_connections=0 "
        "exhausts ephemeral ports via TIME_WAIT (#107880)."
    )


def test_darwin_fallback_branch_updates_transport_reuses_keepalive(monkeypatch):
    """#107880: Darwin getUpdates limits must reach fallback inner transport."""
    monkeypatch.setattr(tg_adapter.sys, "platform", "darwin")
    monkeypatch.delenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", raising=False)
    instances = _drive_connect(
        monkeypatch, proxy_url=None, fallback_ips=["149.154.167.220"]
    )
    limits, transport = _updates_limits_from_fallback(instances)
    assert limits.max_keepalive_connections is not None
    assert limits.max_keepalive_connections >= 1
    asyncio.run(transport.aclose())
    asyncio.run(instances[0].kwargs["httpx_kwargs"]["transport"].aclose())


def test_win32_control_updates_pool_keepalive_is_zero(monkeypatch):
    """#87057: Windows getUpdates pool must still never reuse sockets."""
    monkeypatch.setattr(tg_adapter.sys, "platform", "win32")
    instances = _drive_connect(monkeypatch, proxy_url="http://127.0.0.1:9/")
    limits = _updates_limits_from_proxy(instances)
    assert limits.max_keepalive_connections == 0


def test_linux_control_updates_pool_keepalive_is_zero(monkeypatch):
    """Non-darwin, non-win32 platforms stay at keepalive 0 (fail-open)."""
    monkeypatch.setattr(tg_adapter.sys, "platform", "linux")
    instances = _drive_connect(monkeypatch, proxy_url="http://127.0.0.1:9/")
    limits = _updates_limits_from_proxy(instances)
    assert limits.max_keepalive_connections == 0


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
