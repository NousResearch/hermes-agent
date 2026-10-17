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


    @pytest.mark.parametrize("url", [
        "http://inference-gateway.my-namespace.svc.cluster.local:8080/v1",
        "http://mysvc.default.svc.cluster.local/v1",
        "http://gateway.cluster.local/v1",
        "http://mysvc.default.svc",  # short cluster search-domain form
    ])
    def test_kubernetes_cluster_dns_is_not_local(self, url):
        """*.svc.cluster.local ends in .local but is in-cluster DNS, not mDNS (#138053)."""
        assert is_local_endpoint(url) is False

    @pytest.mark.parametrize("url", [
        "http://nas.local:11434",
        "http://byron.local:11434/v1",
        "http://ollama:11434",  # unqualified Docker Compose / /etc/hosts name
    ])
    def test_mdns_and_unqualified_hosts_still_local(self, url):
        assert is_local_endpoint(url) is True


