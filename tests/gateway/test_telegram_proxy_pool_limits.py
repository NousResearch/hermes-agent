"""Regression tests for the Telegram proxy-path connection-pool leak (#31599).

Behind a flaky local HTTP proxy, httpcore's tunnel path can leave sockets
half-closed on ConnectError. With httpx's default ``keepalive_expiry`` of 5s
those CLOSED sockets accumulate in the general pool faster than they drain,
eventually exhausting the process fd budget after days of operation.

The fix tightens keepalive eviction on the proxy-path HTTPXRequest pools
(reusing the shared #18451 tuning) while preserving the configured
``max_connections`` ceiling so concurrent sends are unaffected.
"""

from __future__ import annotations

import pytest

from gateway.platforms.telegram import _proxy_request_httpx_kwargs


def test_proxy_kwargs_tighten_keepalive_below_httpx_default():
    """keepalive_expiry must be shorter than httpx's 5s default so CLOSED
    sockets through the proxy drain promptly instead of piling up."""
    import httpx

    kwargs = _proxy_request_httpx_kwargs(512)
    limits = kwargs["limits"]
    assert isinstance(limits, httpx.Limits)
    assert limits.keepalive_expiry is not None
    assert limits.keepalive_expiry < 5.0
    assert limits.max_keepalive_connections is not None
    assert 1 <= limits.max_keepalive_connections <= 50


def test_proxy_kwargs_preserve_max_connections_ceiling():
    """The configured connection_pool_size is the max_connections ceiling —
    tightening keepalive must not silently shrink concurrent-send headroom."""
    assert _proxy_request_httpx_kwargs(512)["limits"].max_connections == 512
    assert _proxy_request_httpx_kwargs(64)["limits"].max_connections == 64


def test_proxy_kwargs_empty_when_httpx_unavailable(monkeypatch):
    """If the shared limits helper can't build limits (httpx missing), the
    proxy kwargs are empty so HTTPXRequest falls back to its own limits."""
    import gateway.platforms.telegram as mod

    monkeypatch.setattr(mod, "platform_httpx_limits", lambda: None)
    assert _proxy_request_httpx_kwargs(512) == {}


def test_proxy_kwargs_respect_shared_keepalive_env_override(monkeypatch):
    """The proxy path reuses the shared #18451 tuning, so its env overrides
    apply here too."""
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "3")
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "1.0")
    limits = _proxy_request_httpx_kwargs(512)["limits"]
    assert limits.max_keepalive_connections == 3
    assert limits.keepalive_expiry == 1.0
    # max_connections still tracks the pool size, not the env override.
    assert limits.max_connections == 512
