"""Tests for network.force_ipv4 — the socket.getaddrinfo monkey-patch."""

import importlib
import socket
from unittest.mock import patch, MagicMock

import pytest


def _reload_constants():
    """Reload hermes_constants to get a fresh apply_ipv4_preference."""
    import hermes_constants
    importlib.reload(hermes_constants)
    return hermes_constants


class TestApplyIPv4Preference:
    """Tests for apply_ipv4_preference()."""

    def setup_method(self):
        """Save the original getaddrinfo before each test."""
        self._original = socket.getaddrinfo

    def teardown_method(self):
        """Restore the original getaddrinfo after each test."""
        socket.getaddrinfo = self._original

    def test_noop_when_force_false(self):
        """No patch when force=False."""
        from hermes_constants import apply_ipv4_preference
        original = socket.getaddrinfo
        apply_ipv4_preference(force=False)
        assert socket.getaddrinfo is original

    def test_patches_getaddrinfo_when_forced(self):
        """Patches socket.getaddrinfo when force=True."""
        from hermes_constants import apply_ipv4_preference
        original = socket.getaddrinfo
        apply_ipv4_preference(force=True)
        assert socket.getaddrinfo is not original
        assert getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False) is True

    def test_double_patch_is_safe(self):
        """Calling apply twice doesn't double-wrap."""
        from hermes_constants import apply_ipv4_preference
        apply_ipv4_preference(force=True)
        first_patch = socket.getaddrinfo
        apply_ipv4_preference(force=True)
        assert socket.getaddrinfo is first_patch

    def test_af_unspec_becomes_af_inet(self):
        """AF_UNSPEC (default) calls get rewritten to AF_INET."""
        from hermes_constants import apply_ipv4_preference

        calls = []
        original = socket.getaddrinfo

        def mock_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append(family)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

        socket.getaddrinfo = mock_getaddrinfo
        apply_ipv4_preference(force=True)

        # Call with default family (AF_UNSPEC = 0)
        socket.getaddrinfo("example.com", 80)
        assert calls[-1] == socket.AF_INET, "AF_UNSPEC should be rewritten to AF_INET"

    def test_explicit_family_preserved(self):
        """Explicit AF_INET6 requests are not intercepted."""
        from hermes_constants import apply_ipv4_preference

        calls = []
        original = socket.getaddrinfo

        def mock_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append(family)
            return [(family, socket.SOCK_STREAM, 6, "", ("::1", 80))]

        socket.getaddrinfo = mock_getaddrinfo
        apply_ipv4_preference(force=True)

        socket.getaddrinfo("example.com", 80, family=socket.AF_INET6)
        assert calls[-1] == socket.AF_INET6, "Explicit AF_INET6 should pass through"

    def test_fallback_on_gaierror(self):
        """Falls back to AF_UNSPEC if AF_INET resolution fails."""
        from hermes_constants import apply_ipv4_preference

        call_families = []

        def mock_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            call_families.append(family)
            if family == socket.AF_INET:
                raise socket.gaierror("No A record")
            # AF_UNSPEC fallback returns IPv6
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80))]

        socket.getaddrinfo = mock_getaddrinfo
        apply_ipv4_preference(force=True)

        result = socket.getaddrinfo("ipv6only.example.com", 80)
        # Should have tried AF_INET first, then fallen back to AF_UNSPEC
        assert call_families == [socket.AF_INET, 0]
        assert result[0][0] == socket.AF_INET6


class TestConfigDefault:
    """Verify network section exists in DEFAULT_CONFIG."""

    def test_network_section_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "network" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["network"]["force_ipv4"] is False
