"""Tests for network.force_ipv4 — the socket.getaddrinfo monkey-patch."""

import importlib
import socket



def _reload_constants():
    """Reload hermes_constants to get a fresh apply_ipv4_preference."""
    import hermes_constants
    importlib.reload(hermes_constants)
    return hermes_constants


class TestApplyIPv4Preference:
    """Tests for apply_ipv4_preference()."""

    def setup_method(self):
        """Save the original getaddrinfo + reset the IPv6 probe cache."""
        import hermes_constants
        self._original = socket.getaddrinfo
        hermes_constants._IPV6_ROUTE_ALIVE = None

    def teardown_method(self):
        """Restore the original getaddrinfo + reset the IPv6 probe cache."""
        import hermes_constants
        socket.getaddrinfo = self._original
        hermes_constants._IPV6_ROUTE_ALIVE = None

    def test_noop_when_force_false_and_ipv6_alive(self, monkeypatch):
        """No patch when force=False and the host's IPv6 route is healthy."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "ipv6_route_alive", lambda: True)
        original = socket.getaddrinfo
        hermes_constants.apply_ipv4_preference(force=False)
        assert socket.getaddrinfo is original

    def test_auto_enables_when_ipv6_dead(self, monkeypatch, caplog):
        """force=False auto-patches AND warns when the IPv6 route is dead."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "ipv6_route_alive", lambda: False)
        original = socket.getaddrinfo
        with caplog.at_level("WARNING", logger="hermes_constants"):
            hermes_constants.apply_ipv4_preference(force=False)
        assert socket.getaddrinfo is not original
        assert getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False) is True
        assert any("IPv6 route appears dead" in r.getMessage() for r in caplog.records)

    def test_force_true_skips_the_probe(self, monkeypatch):
        """force=True patches unconditionally, without probing IPv6."""
        import hermes_constants

        def _boom():
            raise AssertionError("ipv6_route_alive must not run when force=True")

        monkeypatch.setattr(hermes_constants, "ipv6_route_alive", _boom)
        hermes_constants.apply_ipv4_preference(force=True)
        assert getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False) is True

    def test_ipv6_route_alive_returns_bool_and_caches(self):
        """The probe returns a bool and caches its result for the process."""
        import hermes_constants
        hermes_constants._IPV6_ROUTE_ALIVE = None
        result = hermes_constants.ipv6_route_alive()
        assert isinstance(result, bool)
        assert hermes_constants._IPV6_ROUTE_ALIVE is result

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
