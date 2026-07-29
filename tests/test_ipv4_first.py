"""Tests for network.ipv4_first — the IPv4-first DNS ordering monkey-patch.

Mirrors the test pattern in test_ipv4_preference.py but for the lighter-weight
apply_ipv6_fallback_ordering() introduced for issue #71215.
"""

import importlib
import socket


def _reload_constants():
    """Reload hermes_constants to get a fresh apply_ipv6_fallback_ordering."""
    import hermes_constants
    importlib.reload(hermes_constants)
    return hermes_constants


class TestApplyIPv6FallbackOrdering:
    """Tests for apply_ipv6_fallback_ordering()."""

    def setup_method(self):
        """Save the original getaddrinfo before each test."""
        self._original = socket.getaddrinfo

    def teardown_method(self):
        """Restore the original getaddrinfo after each test."""
        socket.getaddrinfo = self._original

    def test_noop_when_enabled_false(self):
        """No patch when enabled=False."""
        from hermes_constants import apply_ipv6_fallback_ordering
        original = socket.getaddrinfo
        apply_ipv6_fallback_ordering(enabled=False)
        assert socket.getaddrinfo is original

    def test_patches_getaddrinfo_when_enabled(self):
        """Patches socket.getaddrinfo when enabled=True (default)."""
        from hermes_constants import apply_ipv6_fallback_ordering
        original = socket.getaddrinfo
        apply_ipv6_fallback_ordering()
        assert socket.getaddrinfo is not original
        assert getattr(socket.getaddrinfo, "_hermes_ipv4first_patched", False) is True

    def test_double_patch_is_safe(self):
        """Calling apply twice doesn't double-wrap."""
        from hermes_constants import apply_ipv6_fallback_ordering
        apply_ipv6_fallback_ordering()
        first_patch = socket.getaddrinfo
        apply_ipv6_fallback_ordering()
        assert socket.getaddrinfo is first_patch

    def test_af_unspec_results_sorted_ipv4_first(self):
        """AF_UNSPEC results are sorted so AF_INET comes before AF_INET6."""
        from hermes_constants import apply_ipv6_fallback_ordering

        ipv6_result = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80))
        ipv4_result = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))

        def mock_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            # Return IPv6 first to verify the patch reorders.
            return [ipv6_result, ipv4_result]

        socket.getaddrinfo = mock_getaddrinfo
        apply_ipv6_fallback_ordering()

        result = socket.getaddrinfo("example.com", 80)
        assert result[0][0] == socket.AF_INET, "AF_INET should be sorted first"
        assert result[1][0] == socket.AF_INET6, "AF_INET6 should come after AF_INET"

    def test_explicit_family_preserved(self):
        """Explicit AF_INET6 requests are not reordered (pass-through)."""
        from hermes_constants import apply_ipv6_fallback_ordering

        calls = []

        ipv6_result = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80))

        def mock_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append(family)
            return [ipv6_result]

        socket.getaddrinfo = mock_getaddrinfo
        apply_ipv6_fallback_ordering()

        socket.getaddrinfo("example.com", 80, family=socket.AF_INET6)
        assert calls[-1] == socket.AF_INET6, "Explicit AF_INET6 should pass through"

    def test_ipv6_only_results_preserved(self):
        """Pure-IPv6 hosts still resolve (just in IPv6 order)."""
        from hermes_constants import apply_ipv6_fallback_ordering

        ipv6_result = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 80))

        def mock_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            return [ipv6_result]

        socket.getaddrinfo = mock_getaddrinfo
        apply_ipv6_fallback_ordering()

        result = socket.getaddrinfo("ipv6only.example.com", 80)
        assert len(result) == 1
        assert result[0][0] == socket.AF_INET6


class TestConfigDefault:
    """Verify network.ipv4_first exists in DEFAULT_CONFIG."""

    def test_ipv4_first_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "network" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["network"]["ipv4_first"] is True
