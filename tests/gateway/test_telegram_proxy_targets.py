"""Regression tests for Telegram proxy_targets including custom base_url hostname.

When telegram.extra.base_url is configured (e.g. a self-hosted Bot API server
or Cloudflare Worker proxy), the gateway adapter must include the custom
hostname in proxy_targets so that NO_PROXY entries for it are respected.

Without this, requests to the custom host incorrectly go through the HTTP
proxy even when NO_PROXY lists the hostname.

Issue: #47188
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

from gateway.platforms.base import resolve_proxy_url, should_bypass_proxy


class TestProxyTargetsCustomBaseUrl:
    """Verify that custom base_url hostnames are included in proxy target
    resolution, enabling NO_PROXY bypass for self-hosted Telegram endpoints."""

    def test_no_proxy_bypasses_custom_host_in_targets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When NO_PROXY includes the custom host and it's in target_hosts,
        resolve_proxy_url returns None (proxy bypassed)."""
        custom_host = "tgapi.example.com"
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:8080")
        monkeypatch.setenv("NO_PROXY", custom_host)
        monkeypatch.delenv("no_proxy", raising=False)

        # With custom host in targets → bypass
        result = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org", custom_host])
        assert result is None

    def test_no_proxy_does_not_bypass_without_custom_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When NO_PROXY lists the custom host but it's NOT in target_hosts,
        the proxy is NOT bypassed (the bug scenario)."""
        custom_host = "tgapi.example.com"
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:8080")
        monkeypatch.setenv("NO_PROXY", custom_host)
        monkeypatch.delenv("no_proxy", raising=False)

        # Without custom host in targets → proxy applied
        result = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
        assert result is not None
        assert "proxy.local" in result

    def test_should_bypass_proxy_with_custom_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """should_bypass_proxy returns True when custom host matches NO_PROXY."""
        monkeypatch.setenv("NO_PROXY", "tgapi.example.com")
        monkeypatch.delenv("no_proxy", raising=False)

        assert should_bypass_proxy(["api.telegram.org", "tgapi.example.com"]) is True

    def test_should_bypass_proxy_without_custom_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """should_bypass_proxy returns False when custom host is missing from targets."""
        monkeypatch.setenv("NO_PROXY", "tgapi.example.com")
        monkeypatch.delenv("no_proxy", raising=False)

        assert should_bypass_proxy(["api.telegram.org"]) is False

    def test_custom_host_extraction_from_base_url(self) -> None:
        """Verify hostname extraction from various base_url formats."""
        cases = [
            ("https://tgapi.example.com/bot", "tgapi.example.com"),
            ("https://my-proxy.workers.dev", "my-proxy.workers.dev"),
            ("http://192.168.1.100:8081/bot", "192.168.1.100"),
            ("https://api.telegram.org/bot", "api.telegram.org"),
        ]
        for base_url, expected_host in cases:
            host = urlparse(base_url).hostname
            assert host == expected_host, f"urlparse({base_url!r}).hostname = {host!r}, expected {expected_host!r}"

    def test_wildcard_no_proxy_bypasses_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NO_PROXY=* bypasses proxy for all targets including custom host."""
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:8080")
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.delenv("no_proxy", raising=False)

        result = resolve_proxy_url(
            "TELEGRAM_PROXY",
            target_hosts=["api.telegram.org", "tgapi.example.com"],
        )
        assert result is None
