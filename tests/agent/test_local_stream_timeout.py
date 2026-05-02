"""Tests for local provider stream timeout handling."""

import os
import pytest
from unittest.mock import patch

from agent.model_metadata import is_local_endpoint
from run_agent import AIAgent


def _agent(**kwargs):
    return AIAgent(
        api_key="test-key",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **kwargs,
    )


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
    def test_local_endpoint_keeps_finite_read_timeout(self, base_url):
        """Local endpoint + default timeout -> stays finite, not HERMES_API_TIMEOUT."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            agent = _agent(base_url=base_url)
            assert agent._resolved_stream_read_timeout({}) == 120.0

    def test_user_override_respected_for_local(self):
        """User sets HERMES_STREAM_READ_TIMEOUT -> keep their value even for local."""
        with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "300"}, clear=False):
            agent = _agent(base_url="http://localhost:11434")
            assert agent._resolved_stream_read_timeout({}) == 300.0

    def test_large_output_budget_scales_read_timeout(self):
        """Large generations get more time between chunks without disabling timeout."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            agent = _agent(base_url="http://localhost:11434")
            assert agent._resolved_stream_read_timeout({"max_tokens": 32769}) == 180.0

    def test_local_endpoint_keeps_finite_stale_timeout(self):
        """Local streams still reconnect when no chunks arrive."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            agent = _agent(base_url="http://localhost:11434")
            assert agent._compute_stream_stale_timeout({"messages": []}) == 180.0

    @pytest.mark.parametrize(
        ("max_tokens", "expected"),
        [
            (32769, 300.0),
            (65537, 420.0),
        ],
    )
    def test_large_output_budget_scales_stale_timeout(self, max_tokens, expected):
        """High output budgets get GLM-friendly finite stale thresholds."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            agent = _agent(base_url="http://localhost:11434")
            assert agent._compute_stream_stale_timeout({"messages": [], "max_tokens": max_tokens}) == expected

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
    ])
    def test_remote_endpoint_keeps_default(self, base_url):
        """Remote endpoint -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            agent = _agent(base_url=base_url)
            assert agent._resolved_stream_read_timeout({}) == 120.0

    def test_empty_base_url_keeps_default(self):
        """No base_url set -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            agent = _agent(base_url="https://api.openai.com/v1")
            agent.base_url = ""
            assert agent._resolved_stream_read_timeout({}) == 120.0


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
