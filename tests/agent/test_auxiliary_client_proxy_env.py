"""Regression guard: auxiliary OpenAI clients must use env-only proxy policy.

On macOS, httpx with default ``trust_env=True`` reads system proxy settings
via ``urllib.request.getproxies()`` but not the macOS proxy exception list.
Auxiliary clients (vision, title generation, etc.) must mirror the main
agent: explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars only, via a custom
keepalive transport that suppresses automatic system-proxy detection.
"""
import socket
from unittest.mock import patch

import httpx
import pytest

from agent.auxiliary_client import _create_openai_client, _openai_http_client_kwargs
from agent.process_bootstrap import _get_proxy_for_base_url


@pytest.fixture
def live_proxy_port():
    """A real listening loopback port.

    The transport liveness probe treats a dead loopback proxy as "dial direct", so a test that
    wants the proxy HONORED must point at a port that actually accepts connections — a
    kernel-assigned free port here, not a hardcoded 7897 that only works when Clash is up.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    yield server
    server.close()


def _pool_types(http_client) -> list:
    return [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]


@patch("agent.auxiliary_client.OpenAI")
def test_create_openai_client_routes_via_env_proxy(mock_openai, monkeypatch, live_proxy_port):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{live_proxy_port.getsockname()[1]}")

    _create_openai_client(
        api_key="test-key",
        base_url="https://litellm.internal.example.com/v1",
    )

    http_client = mock_openai.call_args.kwargs.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" in _pool_types(http_client)
    http_client.close()






def test_get_proxy_for_base_url_respects_no_proxy(monkeypatch, live_proxy_port):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    proxy_url = f"http://127.0.0.1:{live_proxy_port.getsockname()[1]}"
    monkeypatch.setenv("HTTPS_PROXY", proxy_url)
    monkeypatch.setenv("NO_PROXY", "internal.example.com")

    assert _get_proxy_for_base_url("https://litellm.internal.example.com/v1") is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == proxy_url


