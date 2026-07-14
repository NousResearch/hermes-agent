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
        "http://ollama:11434",
        "http://litellm:4000/v1",
        "http://hermes-litellm:8080",
        "http://vllm:8000",
    ])
    def test_unqualified_docker_hostnames(self, url):
        """Unqualified hostnames (no dots) are local — Docker Compose, /etc/hosts, etc."""
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


class TestIsKnownLocalInferenceEndpoint:
    """Unit tests for the scoped local-inference detector.

    Port 8080 alone must NOT be treated as inference because it is a
    common cloud-proxy / relay port.  Provider identity (lmstudio, local)
    and unique ports (11434, 1234) are reliable signals.
    """

    def _call(self, base_url, provider=None):
        from agent.chat_completion_helpers import _is_known_local_inference_endpoint
        return _is_known_local_inference_endpoint(base_url, provider)

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434/v1",
        "http://localhost:11434/v1",
    ])
    def test_ollama_port_detected(self, url):
        assert self._call(url) is True

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
    ])
    def test_lm_studio_port_detected(self, url):
        assert self._call(url) is True

    def test_omlx_in_url_detected(self):
        assert self._call("http://127.0.0.1:8080/omlx/v1") is True

    @pytest.mark.parametrize("provider", ["lmstudio", "local"])
    def test_provider_identity_detected(self, provider):
        """Provider lmstudio/local on any local endpoint → True."""
        assert self._call("http://127.0.0.1:8080/v1", provider=provider) is True

    def test_cloud_proxy_on_8080_not_detected(self):
        """Port 8080 without a known provider → NOT inference.

        This is the cloud-proxy regression: a Tailscale relay or HTTP
        CONNECT proxy on :8080 must keep stale detection enabled.
        """
        assert self._call("http://127.0.0.1:8080/v1") is False

    def test_cloud_proxy_on_8080_with_custom_provider_not_detected(self):
        """Provider 'custom' on :8080 → still not inference (could be cloud)."""
        assert self._call("http://127.0.0.1:8080/v1", provider="custom") is False

    def test_remote_endpoint_not_detected(self):
        assert self._call("https://api.openai.com/v1") is False

    def test_empty_base_url_not_detected(self):
        assert self._call(None) is False
        assert self._call("") is False


class TestComputeStreamStaleTimeout:
    """Verify _compute_stream_stale_timeout integrates the scoped detector."""

    def _agent(self, base_url="", provider="custom", model="test-model"):
        from unittest.mock import MagicMock
        a = MagicMock()
        a.base_url = base_url
        a.provider = provider
        a.model = model
        return a

    def test_cloud_proxy_on_8080_keeps_default_timeout(self, monkeypatch):
        """Cloud proxy on :8080 should NOT disable stale detection."""
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        from agent.chat_completion_helpers import _compute_stream_stale_timeout
        agent = self._agent(base_url="http://127.0.0.1:8080/v1", provider="custom")
        result = _compute_stream_stale_timeout(agent, {"model": "test"})
        assert result != float("inf")
        assert result == 180.0  # default

    def test_ollama_disables_stale(self, monkeypatch):
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        from agent.chat_completion_helpers import _compute_stream_stale_timeout
        agent = self._agent(base_url="http://127.0.0.1:11434/v1", provider="custom")
        assert _compute_stream_stale_timeout(agent, {"model": "test"}) == float("inf")

    def test_llamacpp_via_provider_disables_stale(self, monkeypatch):
        """llama.cpp on :8080 with provider 'local' → disabled."""
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        from agent.chat_completion_helpers import _compute_stream_stale_timeout
        agent = self._agent(base_url="http://127.0.0.1:8080/v1", provider="local")
        assert _compute_stream_stale_timeout(agent, {"model": "test"}) == float("inf")

    def test_user_override_keeps_timeout_for_ollama(self, monkeypatch):
        """HERMES_STREAM_STALE_TIMEOUT set → don't auto-disable even for Ollama."""
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "300")
        from agent.chat_completion_helpers import _compute_stream_stale_timeout
        agent = self._agent(base_url="http://127.0.0.1:11434/v1", provider="custom")
        result = _compute_stream_stale_timeout(agent, {"model": "test"})
        assert result == 300.0
