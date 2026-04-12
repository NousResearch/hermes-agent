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


class TestIsLocalEndpointDockerDNS:
    """Verify is_local_endpoint detects Docker/Podman/container DNS names."""

    @pytest.mark.parametrize("base_url", [
        # Unqualified hostnames (Docker Compose service names, /etc/hosts)
        "http://ollama:11434",
        "http://litellm:4000/v1",
        "http://hermes-litellm:4000/v1",
        "http://vllm:8000",
        "http://localai:8080",
        # Well-known Docker/Podman DNS names
        "http://host.docker.internal:11434",
        "http://gateway.docker.internal:8000",
        "http://host.containers.internal:11434",
        "http://kubernetes.docker.internal:6443",
    ])
    def test_container_dns_names_are_local(self, base_url):
        """Docker/Podman DNS names and unqualified hostnames are local."""
        assert is_local_endpoint(base_url) is True

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
        "https://api.anthropic.com",
        "https://api.deepseek.com/v1",
        "https://inference.example.com:8080",
    ])
    def test_remote_fqdn_not_local(self, base_url):
        """Fully-qualified remote hostnames should not be local."""
        # Mock DNS to return a public IP so test doesn't depend on network.
        # Use 8.8.8.8 (a clearly public IP); 203.0.113.x is IANA special-use
        # and Python classifies it as private.
        with patch("socket.gethostbyname", return_value="8.8.8.8"):
            assert is_local_endpoint(base_url) is False

    def test_dns_resolution_fallback(self):
        """Qualified hostname resolving to private IP is local."""
        with patch("socket.gethostbyname", return_value="192.168.1.50"):
            assert is_local_endpoint("http://my-llm.internal:8000") is True

    def test_dns_resolution_public_ip_not_local(self):
        """Qualified hostname resolving to public IP is not local."""
        with patch("socket.gethostbyname", return_value="8.8.8.8"):
            assert is_local_endpoint("http://my-llm.example.com:8000") is False

    def test_dns_resolution_failure_not_local(self):
        """Qualified hostname that fails DNS resolution is not local."""
        import socket
        with patch("socket.gethostbyname", side_effect=socket.gaierror):
            assert is_local_endpoint("http://unknown-host.example.com:8000") is False

    @pytest.mark.parametrize("base_url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://172.17.0.2:8080",
        "http://[::1]:8080",
    ])
    def test_existing_behavior_preserved(self, base_url):
        """Existing local IP and localhost detection still works."""
        assert is_local_endpoint(base_url) is True

    def test_empty_and_none_inputs(self):
        """Edge cases: empty string, None-like."""
        assert is_local_endpoint("") is False
        assert is_local_endpoint("   ") is False
