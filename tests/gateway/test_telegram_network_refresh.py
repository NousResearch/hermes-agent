"""Tests for TelegramFallbackTransport periodic DoH refresh.

Covers the bug fixed in PR: gateway was pinning the same fallback IP
for the entire lifetime of the gateway process even when the local
network conditions changed (ISP reroute, WiFi network change).  The
fix adds a TTL + failure-threshold check that re-runs
discover_fallback_ips() and rebuilds the per-IP transports.
"""

import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.platforms.telegram_network import TelegramFallbackTransport


def _make_transport(ips=None):
    """Construct a transport bypassing the real httpx init."""
    ips = ips or ["1.2.3.4", "5.6.7.8"]
    t = TelegramFallbackTransport.__new__(TelegramFallbackTransport)
    t._fallback_ips = list(ips)
    t._sticky_ip = None
    t._sticky_lock = asyncio.Lock()
    t._last_discovery_at = time.monotonic()
    t._consecutive_connect_failures = 0
    t._discovery_lock = asyncio.Lock()
    t._cached_transport_kwargs = {}
    # Per-IP transport mocks with async aclose
    t._fallbacks = {}
    for ip in ips:
        m = MagicMock()
        m.aclose = AsyncMock()
        t._fallbacks[ip] = m
    return t


class TestMaybeRefreshFallbacks:
    def test_no_refresh_when_failures_below_threshold(self):
        t = _make_transport()
        t._consecutive_connect_failures = 2  # below 3
        t._last_discovery_at = time.monotonic() - 7200  # past TTL
        with patch(
            "gateway.platforms.telegram_network._get_refresh_ttl",
            return_value=3600.0,
        ), patch(
            "gateway.platforms.telegram_network._get_failure_threshold",
            return_value=3,
        ), patch(
            "gateway.platforms.telegram_network.discover_fallback_ips",
            new=AsyncMock(return_value=["9.9.9.9"]),
        ) as m:
            asyncio.run(t._maybe_refresh_fallbacks())
            m.assert_not_called()
            assert t._fallback_ips == ["1.2.3.4", "5.6.7.8"]

    def test_no_refresh_when_ttl_not_elapsed(self):
        t = _make_transport()
        t._consecutive_connect_failures = 5
        t._last_discovery_at = time.monotonic()  # just refreshed
        with patch(
            "gateway.platforms.telegram_network._get_refresh_ttl",
            return_value=3600.0,
        ), patch(
            "gateway.platforms.telegram_network._get_failure_threshold",
            return_value=3,
        ), patch(
            "gateway.platforms.telegram_network.discover_fallback_ips",
            new=AsyncMock(return_value=["9.9.9.9"]),
        ) as m:
            asyncio.run(t._maybe_refresh_fallbacks())
            m.assert_not_called()

    def test_refresh_fires_when_threshold_and_ttl_met(self):
        t = _make_transport(ips=["1.1.1.1"])
        # Capture the OLD transport mocks so we can verify they were closed
        old_aclose_mocks = list(t._fallbacks["1.1.1.1"].aclose.call_args_list)
        t._consecutive_connect_failures = 3
        t._last_discovery_at = time.monotonic() - 7200
        new_ips = ["9.9.9.9", "8.8.8.8"]
        with patch(
            "gateway.platforms.telegram_network._get_refresh_ttl",
            return_value=3600.0,
        ), patch(
            "gateway.platforms.telegram_network._get_failure_threshold",
            return_value=3,
        ), patch(
            "gateway.platforms.telegram_network.discover_fallback_ips",
            new=AsyncMock(return_value=new_ips),
        ), patch.object(
            t, "_transport_kwargs_for", return_value={}
        ):
            asyncio.run(t._maybe_refresh_fallbacks())
            assert t._fallback_ips == new_ips
            assert t._consecutive_connect_failures == 0
            assert "9.9.9.9" in t._fallbacks
            assert "8.8.8.8" in t._fallbacks
            # After refresh, old per-IP transport was replaced
            assert "1.1.1.1" not in t._fallbacks

    def test_refresh_keeps_sticky_if_still_in_list(self):
        t = _make_transport(ips=["1.1.1.1"])
        t._sticky_ip = "5.6.7.8"
        t._consecutive_connect_failures = 3
        t._last_discovery_at = time.monotonic() - 7200
        new_ips = ["5.6.7.8", "9.9.9.9"]
        with patch(
            "gateway.platforms.telegram_network._get_refresh_ttl",
            return_value=3600.0,
        ), patch(
            "gateway.platforms.telegram_network._get_failure_threshold",
            return_value=3,
        ), patch(
            "gateway.platforms.telegram_network.discover_fallback_ips",
            new=AsyncMock(return_value=new_ips),
        ), patch.object(
            t, "_transport_kwargs_for", return_value={}
        ):
            asyncio.run(t._maybe_refresh_fallbacks())
            assert t._sticky_ip == "5.6.7.8"

    def test_refresh_clears_sticky_if_not_in_new_list(self):
        t = _make_transport(ips=["1.1.1.1"])
        t._sticky_ip = "5.6.7.8"  # not in new list
        t._consecutive_connect_failures = 3
        t._last_discovery_at = time.monotonic() - 7200
        new_ips = ["9.9.9.9", "8.8.8.8"]
        with patch(
            "gateway.platforms.telegram_network._get_refresh_ttl",
            return_value=3600.0,
        ), patch(
            "gateway.platforms.telegram_network._get_failure_threshold",
            return_value=3,
        ), patch(
            "gateway.platforms.telegram_network.discover_fallback_ips",
            new=AsyncMock(return_value=new_ips),
        ), patch.object(
            t, "_transport_kwargs_for", return_value={}
        ):
            asyncio.run(t._maybe_refresh_fallbacks())
            assert t._sticky_ip is None

    def test_refresh_swallows_doh_exception(self):
        t = _make_transport()
        original_ips = list(t._fallback_ips)
        t._consecutive_connect_failures = 5
        t._last_discovery_at = time.monotonic() - 7200
        with patch(
            "gateway.platforms.telegram_network._get_refresh_ttl",
            return_value=3600.0,
        ), patch(
            "gateway.platforms.telegram_network._get_failure_threshold",
            return_value=3,
        ), patch(
            "gateway.platforms.telegram_network.discover_fallback_ips",
            new=AsyncMock(side_effect=RuntimeError("DoH unreachable")),
        ):
            # Should NOT raise
            asyncio.run(t._maybe_refresh_fallbacks())
            assert t._consecutive_connect_failures == 0
            assert t._fallback_ips == original_ips
