"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import pytest

from agent.model_metadata import is_local_endpoint


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
        "http://host.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_classic_local_addresses(self, url):
        assert is_local_endpoint(url) is True


    @pytest.mark.parametrize("url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
        "https://evil.docker.internal.example.com",
    ])
    def test_remote_endpoints(self, url):
        assert is_local_endpoint(url) is False

    def test_configured_local_hosts_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_LOCAL_HOSTS", "litellm.homelab.example.com, ai-gateway.internal.example.org")
        assert is_local_endpoint("https://litellm.homelab.example.com:2443/v1") is True
        assert is_local_endpoint("http://ai-gateway.internal.example.org/v1") is True
        assert is_local_endpoint("https://api.openai.com/v1") is False

    def test_configured_local_hosts_config(self, monkeypatch):
        mock_cfg = {
            "agent": {
                "local_hosts": ["custom-llm.homelab.example.com", "local-proxy.internal.example.org"]
            }
        }
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: mock_cfg)
        assert is_local_endpoint("https://custom-llm.homelab.example.com:8443/v1") is True
        assert is_local_endpoint("http://local-proxy.internal.example.org:8000/v1") is True
        assert is_local_endpoint("https://api.anthropic.com/v1") is False


