"""Regression guards for _create_openai_client transport selection.

When #11277 re-landed TCP keepalives, ``_create_openai_client`` began passing
custom ``httpx.Client`` instances with ``HTTPTransport(socket_options=...)``.
That has two important behaviors we want to pin:

1. For regular OpenAI-compatible endpoints, proxy env vars must still be honored.
2. For the ChatGPT Codex backend (``https://chatgpt.com/backend-api/codex``),
   we must NOT inject the custom keepalive transport because it resets the TLS
   handshake (``Connection reset by peer``).
"""
from unittest.mock import patch

import httpx

from run_agent import AIAgent, _get_proxy_from_env


def _make_agent(*, base_url: str, provider: str, model: str):
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        model=model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _extract_http_client(client_kwargs: dict):
    """_create_openai_client calls ``OpenAI(**client_kwargs)``; grab the injected client."""
    return client_kwargs.get("http_client")


def test_get_proxy_from_env_prefers_https_then_http_then_all(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    assert _get_proxy_from_env() is None

    monkeypatch.setenv("ALL_PROXY", "http://all:1")
    assert _get_proxy_from_env() == "http://all:1"

    monkeypatch.setenv("HTTP_PROXY", "http://http:2")
    assert _get_proxy_from_env() == "http://http:2"

    monkeypatch.setenv("HTTPS_PROXY", "http://https:3")
    assert _get_proxy_from_env() == "http://https:3"


def test_get_proxy_from_env_ignores_blank_values(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "   ")
    monkeypatch.setenv("HTTP_PROXY", "http://real-proxy:8080")
    assert _get_proxy_from_env() == "http://real-proxy:8080"


@patch("run_agent.OpenAI")
def test_create_openai_client_routes_via_proxy_when_env_set_for_regular_openai_endpoints(mock_openai, monkeypatch):
    """With HTTPS_PROXY set, non-Codex endpoints should still inject a proxied client."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    agent = _make_agent(
        base_url="https://api.openai.com/v1",
        provider="openai",
        model="gpt-4o-mini",
    )
    kwargs = {
        "api_key": "***",
        "base_url": "https://api.openai.com/v1",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = _extract_http_client(forwarded)
    assert isinstance(http_client, httpx.Client), (
        "Expected _create_openai_client to inject a keepalive-enabled "
        "httpx.Client for regular OpenAI endpoints; got %r" % (http_client,)
    )
    proxied_pools = [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]
    assert "HTTPProxy" in proxied_pools, (
        "Expected httpx.Client to route through HTTPProxy when HTTPS_PROXY is "
        "set; found pools: %r" % (proxied_pools,)
    )
    http_client.close()


@patch("run_agent.OpenAI")
def test_create_openai_client_no_proxy_when_env_unset_for_regular_openai_endpoints(mock_openai, monkeypatch):
    """Without proxy env vars, non-Codex endpoints should still get keepalive transport."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    agent = _make_agent(
        base_url="https://api.openai.com/v1",
        provider="openai",
        model="gpt-4o-mini",
    )
    kwargs = {
        "api_key": "***",
        "base_url": "https://api.openai.com/v1",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = _extract_http_client(forwarded)
    assert isinstance(http_client, httpx.Client)
    pool_types = [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]
    assert "HTTPProxy" not in pool_types, (
        "No proxy env set but httpx.Client still mounted HTTPProxy; "
        "pools were %r" % (pool_types,)
    )
    http_client.close()


@patch("run_agent.OpenAI")
def test_create_openai_client_skips_keepalive_injection_for_codex_provider(mock_openai, monkeypatch):
    """Codex backend must use the default httpx transport to avoid TLS resets."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    agent = _make_agent(
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        model="gpt-5.4",
    )
    kwargs = {
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    assert _extract_http_client(forwarded) is None, (
        "Expected no custom http_client injection for openai-codex; got %r"
        % (forwarded.get("http_client"),)
    )


@patch("run_agent.OpenAI")
def test_create_openai_client_skips_keepalive_injection_for_codex_base_url(mock_openai, monkeypatch):
    """The Codex base URL should bypass keepalive even if provider routing changes."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    agent = _make_agent(
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai",
        model="gpt-5.4",
    )
    kwargs = {
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    assert _extract_http_client(forwarded) is None, (
        "Expected no custom http_client injection for Codex base_url; got %r"
        % (forwarded.get("http_client"),)
    )
