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
        """Save the original getaddrinfo before each test."""
        self._original = socket.getaddrinfo

    def teardown_method(self):
        """Restore the original getaddrinfo after each test."""
        socket.getaddrinfo = self._original


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


class TestEaiAgainFallback:
    """Tests for the gateway's AI_ADDRCONFIG fallback."""

    def setup_method(self):
        self._original = socket.getaddrinfo

    def teardown_method(self):
        socket.getaddrinfo = self._original

    def test_retries_eai_again_without_addrconfig(self):
        from hermes_constants import apply_getaddrinfo_eai_again_fallback

        calls = []
        expected = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

        def degraded_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append((host, port, family, type, proto, flags))
            if flags & socket.AI_ADDRCONFIG:
                raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
            return expected

        socket.getaddrinfo = degraded_getaddrinfo
        apply_getaddrinfo_eai_again_fallback()

        result = socket.getaddrinfo(
            "gateway.test", 443, socket.AF_UNSPEC, socket.SOCK_STREAM, 6,
            socket.AI_ADDRCONFIG | socket.AI_CANONNAME)

        assert result == expected
        assert calls == [
            ("gateway.test", 443, socket.AF_UNSPEC, socket.SOCK_STREAM, 6,
             socket.AI_ADDRCONFIG | socket.AI_CANONNAME),
            ("gateway.test", 443, socket.AF_UNSPEC, socket.SOCK_STREAM, 6,
             socket.AI_CANONNAME),
        ]

    def test_does_not_retry_other_resolution_failures(self):
        from hermes_constants import apply_getaddrinfo_eai_again_fallback

        calls = []

        def missing_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            calls.append(flags)
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        socket.getaddrinfo = missing_getaddrinfo
        apply_getaddrinfo_eai_again_fallback()

        try:
            socket.getaddrinfo("missing.test", 443, flags=socket.AI_ADDRCONFIG)
        except socket.gaierror as exc:
            assert exc.errno == socket.EAI_NONAME
        else:
            raise AssertionError("EAI_NONAME must propagate")
        assert calls == [socket.AI_ADDRCONFIG]



