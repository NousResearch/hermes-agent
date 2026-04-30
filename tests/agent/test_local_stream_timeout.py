"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import os
import socket
import time as _time
import pytest
from unittest.mock import patch

from agent.model_metadata import is_local_endpoint


class TestLocalStreamReadTimeout:
    """Verify stream read timeout auto-detection logic."""

    @pytest.mark.parametrize("base_url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://host.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_local_endpoint_bumps_read_timeout(self, base_url):
        """Local endpoint + default timeout -> bumps to base_timeout."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 1800.0

    def test_user_override_respected_for_local(self):
        """User sets HERMES_STREAM_READ_TIMEOUT -> keep their value even for local."""
        with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "300"}, clear=False):
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = "http://localhost:11434"
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 300.0

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
    ])
    def test_remote_endpoint_keeps_default(self, base_url):
        """Remote endpoint -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0

    def test_empty_base_url_keeps_default(self):
        """No base_url set -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = ""
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0


class TestIsLocalEndpoint:
    """Direct unit tests for is_local_endpoint."""

    @pytest.mark.parametrize("url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://[::1]:11434",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://172.17.0.1:11434",
    ])
    def test_classic_local_addresses(self, url):
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://host.docker.internal:11434",
        "http://host.docker.internal:8080/v1",
        "http://gateway.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_container_dns_names(self, url):
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
        "https://evil.docker.internal.example.com",
    ])
    def test_remote_endpoints(self, url):
        assert is_local_endpoint(url) is False

    @pytest.mark.parametrize("url", [
        "http://100.64.0.0:11434",            # lower bound of CGNAT block
        "http://100.64.0.1:11434/v1",         # lower bound +1
        "http://100.77.243.5:11434",          # representative Tailscale host
        "https://100.100.100.100:443",        # Tailscale MagicDNS anchor
        "https://100.127.255.254:443",        # upper bound -1
        "http://100.127.255.255:11434",       # upper bound of CGNAT block
    ])
    def test_tailscale_cgnat_is_local(self, url):
        """Tailscale 100.64.0.0/10 should be treated as local for timeout bumps."""
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://100.63.255.255:11434",        # just below CGNAT block
        "http://100.128.0.1:11434",           # just above CGNAT block
        "http://100.200.0.1:11434",           # well outside CGNAT
        "http://99.64.0.1:11434",             # first octet wrong
    ])
    def test_near_but_not_cgnat_is_remote(self, url):
        """Hosts adjacent to but outside 100.64.0.0/10 must not match."""
        assert is_local_endpoint(url) is False


class TestIsLocalEndpointPrivateDNS:
    """Reserved private DNS namespaces (.home.arpa, .local, .internal, ...)."""

    def setup_method(self) -> None:
        from agent.model_metadata import _dns_resolution_cache
        _dns_resolution_cache.clear()

    @pytest.mark.parametrize("url", [
        # RFC 8375 — home networks
        "http://ollama.home.arpa/v1",
        "http://ollama.home.arpa:11434",
        "http://server.home.arpa",
        # mDNS (RFC 6762)
        "http://printer.local",
        "http://nas.local:9090",
        # IANA-reserved internal zone
        "http://gitlab.internal:8080",
        # de-facto private conventions
        "http://router.lan",
        "http://wiki.intranet",
        "http://nas.home",
        "http://box.localdomain",
        "http://service.private",
    ])
    def test_private_dns_suffixes_are_local(self, url):
        # Patched so a buggy resolver can't accidentally satisfy this test.
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            side_effect=AssertionError("DNS lookup must not run for private suffixes"),
        ):
            assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        # Suffix-only collision: ".home.arpa" must match at the boundary.
        "https://evil.home.arpa.example.com",
        "https://homearpa.example.com",
        "https://ranch.local.example.com",
    ])
    def test_private_suffix_collision_is_not_local(self, url):
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            side_effect=socket.gaierror("not resolvable"),
        ):
            assert is_local_endpoint(url) is False


class TestIsLocalEndpointDNSResolution:
    """DNS-based fallback for hostnames not covered by literal rules."""

    def setup_method(self) -> None:
        from agent.model_metadata import _dns_resolution_cache
        _dns_resolution_cache.clear()

    @staticmethod
    def _addrinfo_v4(ip: str) -> list:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    @staticmethod
    def _addrinfo_v6(ip: str) -> list:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]

    @pytest.mark.parametrize("ip", [
        "192.168.1.50",     # RFC1918 192.168/16
        "10.0.0.5",         # RFC1918 10/8
        "172.20.0.5",       # RFC1918 172.16/12
        "127.0.0.1",        # loopback
        "169.254.1.5",      # link-local
        "100.77.243.5",     # Tailscale CGNAT
    ])
    def test_hostname_resolving_to_private_ip_is_local(self, ip):
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            return_value=self._addrinfo_v4(ip),
        ):
            assert is_local_endpoint("http://my-internal-box.example.com") is True

    @pytest.mark.parametrize("ip", [
        "8.8.8.8",
        "1.1.1.1",
        "104.16.132.229",
        "199.232.64.140",
    ])
    def test_hostname_resolving_to_public_ip_is_not_local(self, ip):
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            return_value=self._addrinfo_v4(ip),
        ):
            assert is_local_endpoint("http://api.example.com") is False

    def test_dns_resolution_failure_is_not_local(self):
        """NXDOMAIN / unresolvable hosts must not be classified as local."""
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            side_effect=socket.gaierror("nodename nor servname"),
        ):
            assert is_local_endpoint("http://nonexistent.example.com") is False

    def test_slow_resolver_does_not_block_caller(self):
        """If the resolver hangs past _DNS_LOOKUP_TIMEOUT, give up cleanly."""
        from agent.model_metadata import _DNS_LOOKUP_TIMEOUT

        def _slow_lookup(*args, **kwargs):
            _time.sleep(_DNS_LOOKUP_TIMEOUT * 4)
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))]

        start = _time.monotonic()
        with patch(
            "agent.model_metadata.socket.getaddrinfo", side_effect=_slow_lookup,
        ):
            result = is_local_endpoint("http://slow-resolver.example.com")
        elapsed = _time.monotonic() - start

        assert result is False
        # Should return shortly after the timeout, not after the full sleep.
        assert elapsed < _DNS_LOOKUP_TIMEOUT * 3

    def test_resolution_result_is_cached(self):
        """Repeated calls re-use the cached answer, not the resolver."""
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            return_value=self._addrinfo_v4("192.168.1.50"),
        ) as mocked:
            assert is_local_endpoint("http://cached-host.example.com") is True
            assert is_local_endpoint("http://cached-host.example.com") is True
            assert is_local_endpoint("http://cached-host.example.com:9000/v1") is True
        assert mocked.call_count == 1

    def test_ipv6_ula_resolution_is_local(self):
        """IPv6 unique-local addresses (fc00::/7) count as private."""
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            return_value=self._addrinfo_v6("fd12:3456:789a::1"),
        ):
            assert is_local_endpoint("http://my-ula-box.example.com") is True

    def test_ff_tech_home_arpa_endpoint_is_local(self):
        """Regression: ff-tech provider with base_url http://ollama.home.arpa/v1
        must classify as local without needing DNS resolution."""
        with patch(
            "agent.model_metadata.socket.getaddrinfo",
            side_effect=AssertionError("must short-circuit on .home.arpa"),
        ):
            assert is_local_endpoint("http://ollama.home.arpa/v1") is True
