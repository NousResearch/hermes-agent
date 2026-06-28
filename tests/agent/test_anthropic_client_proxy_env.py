"""Regression guard: Anthropic clients must use env-only proxy policy.

On macOS, httpx with default ``trust_env=True`` reads system proxy settings
via ``urllib.request.getproxies()`` but not the macOS proxy exception list.
Anthropic clients (main agent + auxiliary) must mirror the OpenAI fix in
#53702: explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars only, via a custom
keepalive transport that suppresses automatic system-proxy detection.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest


def _pool_types(http_client) -> list:
    return [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    for key in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
        "https_proxy", "http_proxy", "all_proxy",
        "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)


def test_anthropic_client_routes_via_env_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    mock_sdk = MagicMock()
    with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
        from agent.anthropic_adapter import build_anthropic_client
        build_anthropic_client("sk-ant-test-key", base_url="https://api.anthropic.com")

    http_client = mock_sdk.Anthropic.call_args.kwargs.get("http_client")
    assert http_client is not None, "http_client must be injected"
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" in _pool_types(http_client)
    http_client.close()


def test_anthropic_client_no_proxy_when_env_unset():
    mock_sdk = MagicMock()
    with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
        from agent.anthropic_adapter import build_anthropic_client
        build_anthropic_client("sk-ant-test-key", base_url="https://api.anthropic.com")

    http_client = mock_sdk.Anthropic.call_args.kwargs.get("http_client")
    assert http_client is not None, "http_client must be injected"
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" not in _pool_types(http_client)
    http_client.close()


def test_anthropic_client_ignores_macos_system_proxy():
    mock_sdk = MagicMock()
    with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
        with patch(
            "urllib.request.getproxies",
            return_value={"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"},
        ):
            from agent.anthropic_adapter import build_anthropic_client
            build_anthropic_client("sk-ant-test-key", base_url="https://api.anthropic.com")

    http_client = mock_sdk.Anthropic.call_args.kwargs.get("http_client")
    assert http_client is not None, "http_client must be injected"
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" not in _pool_types(http_client)
    http_client.close()
