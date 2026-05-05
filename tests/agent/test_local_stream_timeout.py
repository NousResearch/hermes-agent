"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import os
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

    @pytest.mark.parametrize("url", [
        # mDNS — RFC 6762 reserved
        "http://nas.local:11434",
        "http://printer.local",
        "https://hass.local:8123",
        # Common consumer-router LAN suffixes
        "http://my-server.lan:11434",
        "http://router.home:11434",
        "http://router.home.arpa:8080",   # RFC 8375 homenet
        "http://gateway.internal:11434",
        "http://server.intranet:11434",
        "http://box.localdomain:11434",
        "http://vault.private:11434",
    ])
    def test_private_dns_suffixes_are_local(self, url):
        """mDNS/.lan/.home/.internal etc. resolve only on local
        networks and must be treated as local. Regression for #20346.
        """
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://homeassistant:8123",
        "http://nas:8000",
        "http://ollama-box:11434",
        "http://printer",
    ])
    def test_unqualified_hostnames_are_local(self, url):
        """Bare hostnames with no dots typically resolve via /etc/hosts,
        NetBIOS, or local DHCP — not public DNS — so are private.
        Regression for #20346.
        """
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        # These contain ``.local`` but not as a suffix — must NOT match
        "https://local.example.com",
        "https://my-local.cloud-provider.com",
        # Foo.localhost is loopback per RFC, but our check rejects;
        # we only treat the literal ``localhost`` string. That's the
        # current (intentional) behaviour — this guards against scope
        # creep.
        "https://something-local.io",
        # Multi-label domain that ends in a public suffix lookalike
        "https://api.intranet.example.com",
    ])
    def test_remote_lookalikes_are_not_local(self, url):
        """Domains that contain LAN-suffix-like substrings but are
        actually public must not match. Regression for #20346.
        """
        assert is_local_endpoint(url) is False

    def test_dns_resolution_disabled_by_default(self, monkeypatch):
        """Without HERMES_LOCAL_ENDPOINT_RESOLVE_DNS=1, a hostname that
        is not a private suffix and is not unqualified must not be
        looked up — treat as remote.
        """
        monkeypatch.delenv("HERMES_LOCAL_ENDPOINT_RESOLVE_DNS", raising=False)
        # Intentionally use a public host that we know exists. The
        # function must not call DNS by default; even if it did, the
        # public IP would not be private.
        assert is_local_endpoint("https://example.com") is False

    def test_dns_resolution_opt_in_marks_private_ip_local(self, monkeypatch):
        """With HERMES_LOCAL_ENDPOINT_RESOLVE_DNS=1, a hostname that
        resolves to an RFC-1918 IP must be treated as local. Uses a
        stubbed ``socket.getaddrinfo`` so the test is offline-safe.
        """
        import socket

        monkeypatch.setenv("HERMES_LOCAL_ENDPOINT_RESOLVE_DNS", "1")

        def fake_getaddrinfo(host, port, *args, **kwargs):
            # Pretend ``ollama.example.com`` resolves to 10.0.0.5
            if host == "ollama.example.com":
                return [(2, 1, 6, "", ("10.0.0.5", 0))]
            raise socket.gaierror("not found")

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        assert is_local_endpoint("http://ollama.example.com:11434") is True

    def test_dns_resolution_opt_in_public_ip_remains_remote(self, monkeypatch):
        """With opt-in DNS, a public-DNS hostname that resolves to a
        public IP must still be treated as remote.
        """
        import socket

        monkeypatch.setenv("HERMES_LOCAL_ENDPOINT_RESOLVE_DNS", "1")

        def fake_getaddrinfo(host, port, *args, **kwargs):
            return [(2, 1, 6, "", ("8.8.8.8", 0))]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        assert is_local_endpoint("https://api.example.com") is False
