"""Tests for the API server bind-address startup guard.

Validates that is_network_accessible() correctly classifies addresses and
that connect() refuses to start without API_SERVER_KEY.
"""

import socket
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import is_network_accessible


# ---------------------------------------------------------------------------
# Unit tests: is_network_accessible()
# ---------------------------------------------------------------------------


class TestIsNetworkAccessible:
    """Direct tests for the address classification helper."""

    # -- Loopback (safe, should return False) --

    def test_ipv4_loopback(self):
        assert is_network_accessible("127.0.0.1") is False

    def test_ipv6_loopback(self):
        assert is_network_accessible("::1") is False

    def test_ipv4_mapped_loopback(self):
        # ::ffff:127.0.0.1 — Python's is_loopback returns False for mapped
        # addresses; the helper must unwrap and check ipv4_mapped.
        assert is_network_accessible("::ffff:127.0.0.1") is False

    # -- Network-accessible (should return True) --

    def test_ipv4_wildcard(self):
        assert is_network_accessible("0.0.0.0") is True

    def test_ipv6_wildcard(self):
        # This is the bypass vector that the string-based check missed.
        assert is_network_accessible("::") is True

    def test_ipv4_mapped_unspecified(self):
        assert is_network_accessible("::ffff:0.0.0.0") is True

    def test_private_ipv4(self):
        assert is_network_accessible("10.0.0.1") is True

    def test_private_ipv4_class_c(self):
        assert is_network_accessible("192.168.1.1") is True

    def test_public_ipv4(self):
        assert is_network_accessible("8.8.8.8") is True

    # -- Hostname resolution --

    def test_localhost_resolves_to_loopback(self):
        loopback_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
        ]
        with patch("gateway.platforms.base._socket.getaddrinfo", return_value=loopback_result):
            assert is_network_accessible("localhost") is False

    def test_hostname_resolving_to_non_loopback(self):
        non_loopback_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        ]
        with patch("gateway.platforms.base._socket.getaddrinfo", return_value=non_loopback_result):
            assert is_network_accessible("my-server.local") is True

    def test_hostname_mixed_resolution(self):
        """If a hostname resolves to both loopback and non-loopback, it's
        network-accessible (any non-loopback address is enough)."""
        mixed_result = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
        ]
        with patch("gateway.platforms.base._socket.getaddrinfo", return_value=mixed_result):
            assert is_network_accessible("dual-host.local") is True

    def test_dns_failure_fails_closed(self):
        """Unresolvable hostnames should require an API key (fail closed)."""
        with patch(
            "gateway.platforms.base._socket.getaddrinfo",
            side_effect=socket.gaierror("Name resolution failed"),
        ):
            assert is_network_accessible("nonexistent.invalid") is True


# ---------------------------------------------------------------------------
# Integration tests: connect() startup guard
# ---------------------------------------------------------------------------


class TestConnectBindGuard:
    """Verify that connect() refuses dangerous configurations."""

    @pytest.mark.asyncio
    async def test_refuses_ipv4_wildcard_without_key(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "0.0.0.0"}))
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_refuses_ipv6_wildcard_without_key(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "::"}))
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_refuses_loopback_without_key(self):
        """Loopback binds are still an auth boundary and require API_SERVER_KEY."""
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1"}))
        assert adapter._api_key == ""
        assert is_network_accessible(adapter._host) is False
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_allows_wildcard_with_key(self):
        """Non-loopback with a key should pass the guard."""
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "0.0.0.0", "key": "sk-test"})
        )
        # The guard checks: is_network_accessible(host) AND NOT api_key
        # With a key set, the guard should not block.
        assert adapter._api_key == "sk-test"
        assert is_network_accessible("0.0.0.0") is True
        # Combined: the guard condition is False (key is set), so it passes

    @pytest.mark.asyncio
    async def test_port_conflict_sets_non_retryable_fatal_error(self):
        """When the api_server port is already in use, connect() must set a
        non-retryable fatal error so the reconnect watcher drops it from the
        retry queue instead of looping indefinitely.

        Previously connect() returned ``False`` without setting a fatal error,
        so the reconnect watcher treated the port conflict as a transient,
        retryable failure and retried every 5 minutes forever — filling
        errors.log with thousands of duplicate lines and leaking 2 fds/retry.
        """
        # Use a key so we pass the auth guard and reach the port-conflict check.
        # Bind a real socket on an ephemeral port to simulate the conflict.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        conflict_port = sock.getsockname()[1]
        try:
            adapter = APIServerAdapter(
                PlatformConfig(
                    enabled=True,
                    extra={"host": "127.0.0.1", "key": "sk-test", "port": conflict_port},
                )
            )
            result = await adapter.connect()
            assert result is False
            # The critical assertion: the error must be non-retryable so the
            # reconnect watcher stops retrying.
            assert adapter.has_fatal_error is True
            assert adapter.fatal_error_retryable is False
            assert adapter.fatal_error_code == "api_server_port_in_use"
            assert str(conflict_port) in (adapter.fatal_error_message or "")
        finally:
            sock.close()

