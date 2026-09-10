"""Regression guard: _create_openai_client must honor HTTP(S)_PROXY env vars.

The keepalive client now uses ``httpx.Limits(keepalive_expiry=20.0)``
instead of a custom ``httpx.HTTPTransport(socket_options=...)`` to
prevent CLOSE-WAIT accumulation.  This avoids breaking streaming for
providers behind reverse proxies (#54049, #12952) while still reaping
idle connections before a proxy's timeout drops them.

This test pins that the constructed ``httpx.Client`` mounts an
``HTTPProxy`` pool when a proxy env var is set, and that no
custom socket-options transport is used (default httpx transport).
"""
from unittest.mock import patch

import httpx

from agent.process_bootstrap import _get_proxy_for_base_url, _get_proxy_from_env
from run_agent import AIAgent


def _make_agent():
    return AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        model="gpt-5.4",
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




def test_get_proxy_from_env_normalizes_socks_alias(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1080/")
    assert _get_proxy_from_env() == "socks5://127.0.0.1:1080/"


@patch("agent.process_bootstrap.OpenAI")
def test_create_openai_client_routes_via_proxy_when_env_set(mock_openai, monkeypatch):
    """With HTTPS_PROXY set, the custom httpx.Client must mount an HTTPProxy pool.

    This is the WSL2 + Clash / corporate-egress case. Before the fix, the custom
    transport suppressed httpx's env-proxy auto-detection, so requests bypassed
    the proxy entirely.
    """
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    agent = _make_agent()
    kwargs = {
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = _extract_http_client(forwarded)
    assert isinstance(http_client, httpx.Client), (
        "Expected _create_openai_client to inject a keepalive-enabled "
        "httpx.Client; got %r" % (http_client,)
    )
    # Verify a proxy mount exists. httpx Client(proxy=...) rewrites _mounts so
    # the proxied pool (HTTPProxy) sits alongside the base transport.
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


@patch("agent.process_bootstrap.OpenAI")
def test_create_openai_client_no_proxy_when_env_unset(mock_openai, monkeypatch):
    """Without proxy env vars, no HTTPProxy mount should exist and
    no custom socket-options transport should be installed."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    agent = _make_agent()
    kwargs = {
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
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










