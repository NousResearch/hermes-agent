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


@patch("agent.auxiliary_client.OpenAI")
def test_create_openai_client_no_proxy_when_env_unset(mock_openai, monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)

    _create_openai_client(
        api_key="test-key",
        base_url="https://litellm.internal.example.com/v1",
    )

    http_client = mock_openai.call_args.kwargs.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" not in _pool_types(http_client)
    http_client.close()


@patch("agent.auxiliary_client.OpenAI")
def test_create_openai_client_ignores_macos_system_proxy(mock_openai, monkeypatch):
    """System proxy from getproxies() must not apply when env vars are unset."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)

    with patch(
        "urllib.request.getproxies",
        return_value={"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"},
    ):
        _create_openai_client(
            api_key="test-key",
            base_url="https://litellm.internal.example.com/v1",
        )

    http_client = mock_openai.call_args.kwargs.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" not in _pool_types(http_client)
    http_client.close()


def test_get_proxy_for_base_url_respects_no_proxy(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "internal.example.com")

    assert _get_proxy_for_base_url("https://litellm.internal.example.com/v1") is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://127.0.0.1:7897"


def test_openai_http_client_kwargs_async_mode():
    kwargs = _openai_http_client_kwargs(
        "https://litellm.internal.example.com/v1",
        async_mode=True,
    )
    assert isinstance(kwargs["http_client"], httpx.AsyncClient)


# ── no_proxy CIDR matching ──────────────────────────────────────────────

def test_no_proxy_cidr_bypass(monkeypatch):
    from agent.process_bootstrap import _get_proxy_for_base_url

    for key in ("HTTPS_PROXY", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,10.0.0.0/8")

    # 10.0.0.0/8 includes private-range IPs like 10.10.10.101
    # socket.getaddrinfo won't resolve random names, so test with a
    # host that is already an IP — CIDR matching works on IPs directly.
    assert _get_proxy_for_base_url("https://10.10.10.101/v1") is None


def test_no_proxy_cidr_no_match(monkeypatch):
    from agent.process_bootstrap import _get_proxy_for_base_url

    for key in ("HTTPS_PROXY", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")

    # 1.2.3.4 is outside the excluded subnet — proxy should apply.
    assert _get_proxy_for_base_url("https://1.2.3.4/v1") == "http://127.0.0.1:7897"


def test_no_proxy_cidr_invalid_entry_ignored(monkeypatch):
    from agent.process_bootstrap import _get_proxy_for_base_url

    for key in ("HTTPS_PROXY", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "not-a-cidr,not/a/network,192.168.1.0/24")

    # Invalid entries are ignored; 192.168.1.0/24 still works.
    assert _get_proxy_for_base_url("https://192.168.1.50/v1") is None
