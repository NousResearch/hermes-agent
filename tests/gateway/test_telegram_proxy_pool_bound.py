"""Tests for the bounded proxy connection pool (issue #31599).

When Telegram runs behind a flaky local HTTP proxy, half-closed sockets
accumulate in the general request pool faster than httpx evicts them and
eventually exhaust the process fd limit. The fix bounds the pool on the
proxy path via ``httpx.Limits`` so the leak is capped and surfaces
immediately instead of wedging the gateway after ~2 days.
"""

from __future__ import annotations

import httpx

from gateway.platforms.telegram import _bounded_proxy_limits


def test_default_caps_are_bounded():
    limits = _bounded_proxy_limits()
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 20
    assert limits.max_keepalive_connections == 10


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_PROXY_MAX_CONNECTIONS", "40")
    monkeypatch.setenv("HERMES_TELEGRAM_PROXY_MAX_KEEPALIVE", "5")
    limits = _bounded_proxy_limits()
    assert limits.max_connections == 40
    assert limits.max_keepalive_connections == 5


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HERMES_TELEGRAM_PROXY_MAX_CONNECTIONS", "not-an-int")
    limits = _bounded_proxy_limits()
    assert limits.max_connections == 20
    assert limits.max_keepalive_connections == 10
