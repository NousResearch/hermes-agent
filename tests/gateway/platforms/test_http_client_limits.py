"""Tests for gateway.platforms._http_client_limits — platform_httpx_limits()."""

from __future__ import annotations

import httpx
import pytest

from gateway.platforms._http_client_limits import (
    _DEFAULT_KEEPALIVE_EXPIRY_S,
    _DEFAULT_MAX_KEEPALIVE,
    platform_httpx_limits,
)


# ============================================================================
# platform_httpx_limits
# ============================================================================
class TestPlatformHttpxLimits:
    def test_returns_limits_with_defaults(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)
        limits = platform_httpx_limits()
        assert limits is not None
        assert limits.keepalive_expiry == _DEFAULT_KEEPALIVE_EXPIRY_S
        assert limits.max_keepalive_connections == _DEFAULT_MAX_KEEPALIVE

    def test_custom_keepalive_expiry(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "5.0")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == 5.0

    def test_custom_max_keepalive(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "20")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
        limits = platform_httpx_limits()
        assert limits.max_keepalive_connections == 20

    def test_both_custom(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "3.5")
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "15")
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == 3.5
        assert limits.max_keepalive_connections == 15

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "")
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "")
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == _DEFAULT_KEEPALIVE_EXPIRY_S
        assert limits.max_keepalive_connections == _DEFAULT_MAX_KEEPALIVE

    def test_whitespace_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "   ")
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "   ")
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == _DEFAULT_KEEPALIVE_EXPIRY_S
        assert limits.max_keepalive_connections == _DEFAULT_MAX_KEEPALIVE

    def test_invalid_float_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "not-float")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == _DEFAULT_KEEPALIVE_EXPIRY_S

    def test_invalid_int_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "not-int")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
        limits = platform_httpx_limits()
        assert limits.max_keepalive_connections == _DEFAULT_MAX_KEEPALIVE

    def test_zero_expiry_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "0")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == _DEFAULT_KEEPALIVE_EXPIRY_S

    def test_negative_expiry_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "-1")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == _DEFAULT_KEEPALIVE_EXPIRY_S

    def test_zero_max_keepalive_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "0")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
        limits = platform_httpx_limits()
        assert limits.max_keepalive_connections == _DEFAULT_MAX_KEEPALIVE

    def test_negative_max_keepalive_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "-5")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
        limits = platform_httpx_limits()
        assert limits.max_keepalive_connections == _DEFAULT_MAX_KEEPALIVE

    def test_small_positive_expiry(self, monkeypatch):
        monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "0.1")
        monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)
        limits = platform_httpx_limits()
        assert limits.keepalive_expiry == 0.1


# ============================================================================
# Default constants
# ============================================================================
class TestDefaultConstants:
    def test_keepalive_expiry_default(self):
        assert _DEFAULT_KEEPALIVE_EXPIRY_S == 2.0

    def test_max_keepalive_default(self):
        assert _DEFAULT_MAX_KEEPALIVE == 10
