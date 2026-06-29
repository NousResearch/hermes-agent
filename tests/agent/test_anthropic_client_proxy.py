"""Regression guard: build_anthropic_client must not pick up the OS system proxy.

The Anthropic SDK's default httpx client uses ``trust_env=True``. On macOS (and
Windows) httpx then resolves proxies via ``urllib.request.getproxies()``, which
includes the *system* proxy (``scutil --proxy`` on macOS) — not just the
``*_PROXY`` env vars. When a system proxy is configured (e.g. Clash/mihomo at
``127.0.0.1:7890``) but no proxy env var is set, requests to an endpoint the
proxy cannot reach (e.g. an intranet Anthropic-compatible gateway) get tunneled
through the proxy and fail with an opaque ``APIConnectionError`` /
``ConnectError [SSL: UNEXPECTED_EOF_WHILE_READING]``.

The OpenAI-wire path (``_build_keepalive_http_client`` in ``run_agent.py``) is
already immune because it builds a custom ``httpx.Client`` (which disables
httpx's ``trust_env`` proxy auto-detection) and routes only through hermes's
own env-/NO_PROXY-aware resolver. ``build_anthropic_client`` now mirrors that:
it injects an ``httpx.Client(trust_env=False, proxy=_get_proxy_for_base_url(...))``
so the system proxy is never silently applied, while explicit HTTP(S)_PROXY env
still works and NO_PROXY is honored.

Tracked in NousResearch/hermes-agent#25319 (the httpx-path sibling of the
already-fixed OpenAI/auxiliary cases #14451 / #25319 loopback fixes).
"""
import types

import httpx

import agent.anthropic_adapter as aa


_PROXY_ENV_KEYS = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
    "https_proxy", "http_proxy", "all_proxy",
    "NO_PROXY", "no_proxy",
)


def _clear_proxy_env(monkeypatch):
    for key in _PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _capture_anthropic_kwargs(monkeypatch):
    """Stub the lazily-imported anthropic SDK so no real client/network is needed."""
    captured = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        aa, "_get_anthropic_sdk",
        lambda: types.SimpleNamespace(Anthropic=_FakeAnthropic),
    )
    return captured


def _proxy_pool_names(http_client):
    return [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]


def test_anthropic_client_disables_trust_env(monkeypatch):
    """With no proxy env vars, the injected client must have trust_env=False so
    the OS system proxy (getproxies()) is never consulted, and no proxy pool is
    mounted."""
    _clear_proxy_env(monkeypatch)
    captured = _capture_anthropic_kwargs(monkeypatch)

    aa.build_anthropic_client(
        "sk-test", base_url="https://gateway.internal.example.com/anthropic"
    )

    http_client = captured["kwargs"].get("http_client")
    assert isinstance(http_client, httpx.Client), (
        "build_anthropic_client must inject its own httpx.Client; got %r"
        % (http_client,)
    )
    assert http_client._trust_env is False, (
        "trust_env must be False so the SDK's httpx never reads the OS system "
        "proxy via urllib.getproxies()"
    )
    assert "HTTPProxy" not in _proxy_pool_names(http_client)
    http_client.close()


def test_anthropic_client_honors_env_proxy(monkeypatch):
    """An explicit HTTPS_PROXY env var must still route a remote endpoint through
    the proxy (egress users are unaffected)."""
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    captured = _capture_anthropic_kwargs(monkeypatch)

    aa.build_anthropic_client("sk-test", base_url="https://api.anthropic.com")

    http_client = captured["kwargs"]["http_client"]
    assert "HTTPProxy" in _proxy_pool_names(http_client), (
        "Explicit HTTPS_PROXY must still be honored for remote endpoints"
    )
    http_client.close()


def test_anthropic_client_honors_no_proxy(monkeypatch):
    """NO_PROXY must suppress the proxy for a matching host even when HTTPS_PROXY
    is set."""
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "internal.example.com")
    captured = _capture_anthropic_kwargs(monkeypatch)

    aa.build_anthropic_client(
        "sk-test", base_url="https://gw.internal.example.com/anthropic"
    )

    http_client = captured["kwargs"]["http_client"]
    assert "HTTPProxy" not in _proxy_pool_names(http_client), (
        "NO_PROXY host must not route through HTTPProxy"
    )
    http_client.close()
