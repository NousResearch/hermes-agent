"""Regression guard: auxiliary OpenAI clients must use env-only proxy policy.

On macOS, httpx with default ``trust_env=True`` reads system proxy settings
via ``urllib.request.getproxies()`` but not the macOS proxy exception list.
Auxiliary clients (vision, title generation, etc.) must mirror the main
agent: explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars only, via a custom
keepalive transport that suppresses automatic system-proxy detection.
"""
from unittest.mock import patch

import httpx

from agent.auxiliary_client import _create_openai_client, _openai_http_client_kwargs
from agent.process_bootstrap import _get_proxy_for_base_url


def _pool_types(http_client) -> list:
    return [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]


@patch("agent.auxiliary_client.OpenAI")
def test_create_openai_client_routes_via_env_proxy(mock_openai, monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    _create_openai_client(
        api_key="test-key",
        base_url="https://litellm.internal.example.com/v1",
    )

    http_client = mock_openai.call_args.kwargs.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" in _pool_types(http_client)
    http_client.close()






def test_get_proxy_for_base_url_respects_no_proxy(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "internal.example.com")

    assert _get_proxy_for_base_url("https://litellm.internal.example.com/v1") is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://127.0.0.1:7897"


def test_get_proxy_for_base_url_honors_cidr_no_proxy(monkeypatch):
    """CIDR NO_PROXY entries must bypass the proxy, exactly as curl does.

    urllib.request.proxy_bypass_environment() only understands exact hosts, `.domain`
    suffixes and `*`, so `NO_PROXY=127.0.0.0/8,10.0.0.0/8,192.168.0.0/16` (the shape
    Clash/v2ray exporters write) silently sent every local/LAN Ollama request to the
    proxy while curl went direct (#101803).
    """
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "127.0.0.0/8,10.0.0.0/8,192.168.0.0/16")

    assert _get_proxy_for_base_url("http://127.0.0.1:11434/v1") is None
    assert _get_proxy_for_base_url("http://192.168.1.50:11434/v1") is None
    assert _get_proxy_for_base_url("http://10.4.0.9:11434/v1") is None
    # ...and a real remote host still goes through the proxy.
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://127.0.0.1:7897"


def test_get_proxy_for_base_url_honors_wildcard_and_host_port_entries(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "*.lan.example,ollama-box:11434")

    assert _get_proxy_for_base_url("http://gpu.lan.example:11434/v1") is None
    assert _get_proxy_for_base_url("http://ollama-box:11434/v1") is None
    # A host:port entry is port-specific, and an unrelated host stays proxied.
    assert _get_proxy_for_base_url("http://ollama-box:9999/v1") == "http://127.0.0.1:7897"
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://127.0.0.1:7897"


