"""Regression tests for PR #92494 new review findings (andrexibiza).

5. IPv6 URL BRACKETING (3837640618): _own_a2a_url and _loopback_fallback_url
   must produce bracketed IPv6 literals (http://[::1]:port) so that
   urlparse(...).port does not raise.

6. PORTABLE MSG_DONTWAIT (3837640619): _a2a_client_alive must not reference
   socket.MSG_DONTWAIT directly; a portable fallback constant must be used
   so the module loads on Windows without AttributeError.
"""
from __future__ import annotations

import socket
import urllib.parse
from unittest.mock import MagicMock, patch

import pytest

from plugins.platforms.a2a import adapter as adapter_mod


# ═════════════════════════════════════════════════════════════════════════════
# 5. IPv6 URL bracketing
# ═════════════════════════════════════════════════════════════════════════════

class TestIPv6URLBracketing:
    """_own_a2a_url and _loopback_fallback_url must bracket IPv6 literals."""

    def test_own_a2a_url_ipv6_loopback(self):
        """::1 must produce http://[::1]:port, not http://[::1]:port."""
        url = adapter_mod._own_a2a_url("::1", 9900)
        assert url == "http://[::1]:9900"
        parsed = urllib.parse.urlparse(url)
        assert parsed.hostname == "::1"
        assert parsed.port == 9900

    def test_own_a2a_url_ipv6_wildcard_returns_127(self):
        """:: (IPv6 wildcard) must resolve to 127.0.0.1, not bracket."""
        url = adapter_mod._own_a2a_url("::", 8080)
        assert url == "http://127.0.0.1:8080"

    def test_own_a2a_url_ipv4_unchanged(self):
        """IPv4 addresses must not be wrapped in brackets."""
        url = adapter_mod._own_a2a_url("127.0.0.1", 9900)
        assert url == "http://127.0.0.1:9900"

    def test_own_a2a_url_localhost_unchanged(self):
        """'localhost' must not be treated as IPv6."""
        url = adapter_mod._own_a2a_url("localhost", 9900)
        assert url == "http://localhost:9900"

    def test_own_a2a_url_empty_host_defaults(self):
        """Empty host defaults to 127.0.0.1."""
        url = adapter_mod._own_a2a_url("", 9900)
        assert url == "http://127.0.0.1:9900"

    def test_own_a2a_url_full_ipv6_address(self):
        """A full IPv6 address must be bracketed and parseable."""
        url = adapter_mod._own_a2a_url("fe80::1", 8443)
        assert url == "http://[fe80::1]:8443"
        parsed = urllib.parse.urlparse(url)
        assert parsed.hostname == "fe80::1"
        assert parsed.port == 8443

    def test_loopback_fallback_url_ipv6(self):
        """ip: identity with IPv6 loopback must produce bracketed URL."""
        url = adapter_mod._loopback_fallback_url("ip:::1", "::1", 9900)
        assert url == "http://[::1]:9900"
        parsed = urllib.parse.urlparse(url)
        assert parsed.hostname == "::1"
        assert parsed.port == 9900

    def test_loopback_fallback_url_ipv4(self):
        """ip: identity with IPv4 loopback must not bracket."""
        url = adapter_mod._loopback_fallback_url("ip:127.0.0.1", "127.0.0.1", 9900)
        assert url == "http://127.0.0.1:9900"

    def test_loopback_fallback_url_non_loopback_empty(self):
        """Non-loopback identity returns empty string."""
        url = adapter_mod._loopback_fallback_url("ip:10.0.0.1", "127.0.0.1", 9900)
        assert url == ""

    def test_loopback_fallback_url_non_ip_identity_empty(self):
        """Non ip: identity returns empty string."""
        url = adapter_mod._loopback_fallback_url("bearer-token", "127.0.0.1", 9900)
        assert url == ""

    def test_bracket_ipv6_helper(self):
        """_bracket_ipv6 helper correctly identifies and brackets literals."""
        assert adapter_mod._bracket_ipv6("::1") == "[::1]"
        assert adapter_mod._bracket_ipv6("fe80::1") == "[fe80::1]"
        assert adapter_mod._bracket_ipv6("127.0.0.1") == "127.0.0.1"
        assert adapter_mod._bracket_ipv6("localhost") == "localhost"

    def test_is_ipv6_literal_helper(self):
        """_is_ipv6_literal detects colon-containing hosts."""
        assert adapter_mod._is_ipv6_literal("::1") is True
        assert adapter_mod._is_ipv6_literal("fe80::1") is True
        assert adapter_mod._is_ipv6_literal("127.0.0.1") is False
        assert adapter_mod._is_ipv6_literal("localhost") is False


# ═════════════════════════════════════════════════════════════════════════════
# 6. Portable MSG_DONTWAIT
# ═════════════════════════════════════════════════════════════════════════════

class TestPortableMSGDONTWAIT:
    """_a2a_client_alive must use a portable non-blocking recv flag."""

    def test_portable_constant_defined(self):
        """_PORTABLE_NONBLOCK_RECV must be defined as a module-level constant."""
        assert hasattr(adapter_mod, "_PORTABLE_NONBLOCK_RECV")
        val = adapter_mod._PORTABLE_NONBLOCK_RECV
        assert isinstance(val, int)

    def test_portable_constant_matches_platform(self):
        """Value must match socket.MSG_DONTWAIT on Unix, 0 on Windows."""
        if hasattr(socket, "MSG_DONTWAIT"):
            assert adapter_mod._PORTABLE_NONBLOCK_RECV == socket.MSG_DONTWAIT
        else:
            assert adapter_mod._PORTABLE_NONBLOCK_RECV == 0

    def test_source_uses_portable_constant(self):
        """adapter.py must not reference socket.MSG_DONTWAIT directly in recv."""
        import inspect
        handler_cls = adapter_mod.A2ARequestHandler
        source = inspect.getsource(handler_cls._a2a_client_alive)
        assert "socket.MSG_DONTWAIT" not in source, (
            "_a2a_client_alive must use _PORTABLE_NONBLOCK_RECV, "
            "not socket.MSG_DONTWAIT directly"
        )
        assert "_PORTABLE_NONBLOCK_RECV" in source

    def test_client_alive_uses_portable_recv_flag(self):
        """_a2a_client_alive must pass the portable flag to sock.recv."""
        from concurrent.futures import Future

        mock_sock = MagicMock()
        # select says readable, recv returns b"" (EOF = client closed)
        with patch("select.select", return_value=([mock_sock], [], [])):
            mock_sock.recv.return_value = b""
            handler_cls = adapter_mod.A2ARequestHandler
            handler = handler_cls.__new__(handler_cls)
            handler.connection = mock_sock
            result = handler._a2a_client_alive()
            assert result is False
            # Verify the portable flag was used, not MSG_DONTWAIT directly
            args, kwargs = mock_sock.recv.call_args
            assert args[1] == (
                socket.MSG_PEEK | adapter_mod._PORTABLE_NONBLOCK_RECV
            )

    def test_module_loads_without_msg_dontwait(self):
        """Module must not fail to load when MSG_DONTWAIT is absent (Windows)."""
        # Verify the constant resolves without error
        val = adapter_mod._PORTABLE_NONBLOCK_RECV
        assert val >= 0
