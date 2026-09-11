"""Behavior contracts for immutable runtime resolution and client bundles."""

from types import MappingProxyType

import pytest

from agent.runtime_bundle import ResolvedRuntime, build_client_bundle


def test_resolved_runtime_is_immutable_mapping():
    source = {
        "provider": "custom",
        "model": "btc-model",
        "api_mode": "chat_completions",
        "api_key": "secret",
        "base_url": "https://btc.example/v1",
        "extra_headers": {"CF-Access-Client-Id": "client-id"},
        "request_overrides": {"extra_body": {"route": "btc"}},
    }

    runtime = ResolvedRuntime.from_mapping(source)
    source["extra_headers"]["CF-Access-Client-Id"] = "mutated"

    assert runtime["provider"] == "custom"
    assert runtime.extra_headers["CF-Access-Client-Id"] == "client-id"
    assert isinstance(runtime.extra_headers, MappingProxyType)
    with pytest.raises(TypeError):
        runtime.extra_headers["new"] = "value"
    with pytest.raises(TypeError):
        runtime["request_overrides"]["new"] = "value"


def test_openai_wire_bundle_passes_runtime_headers_and_tls_to_builder():
    observed = {}
    client = object()

    def builder(kwargs):
        observed.update(kwargs)
        return client

    runtime = ResolvedRuntime.from_mapping(
        {
            "provider": "custom",
            "model": "btc-model",
            "api_mode": "codex_responses",
            "api_key": "secret",
            "base_url": "https://btc.example/v1",
            "default_headers": {
                "User-Agent": "HermesAgent/test",
                "CF-Access-Client-Secret": "stale-default",
            },
            "extra_headers": {"CF-Access-Client-Secret": "service-secret"},
            "ssl_verify": False,
            "default_query": {"api-version": "2026-08-01"},
        }
    )

    bundle = build_client_bundle(runtime, openai_builder=builder, timeout=17)

    assert bundle.client is client
    assert observed["default_headers"]["CF-Access-Client-Secret"] == "service-secret"
    assert observed["default_headers"]["User-Agent"] == "HermesAgent/test"
    assert observed["ssl_verify"] is False
    assert observed["timeout"] == 17
    assert observed["default_query"] == {"api-version": "2026-08-01"}


def test_anthropic_wire_bundle_merges_runtime_headers_through_builder():
    observed = {}
    client = object()

    def builder(api_key, base_url, **kwargs):
        observed.update(api_key=api_key, base_url=base_url, **kwargs)
        return client

    runtime = ResolvedRuntime.from_mapping(
        {
            "provider": "custom",
            "model": "claude-proxy",
            "api_mode": "anthropic_messages",
            "api_key": "secret",
            "base_url": "https://btc.example/anthropic",
            "extra_headers": {"CF-Access-Client-Id": "client-id"},
        }
    )

    bundle = build_client_bundle(runtime, anthropic_builder=builder, timeout=23)

    assert bundle.anthropic_client is client
    assert observed["default_headers"]["CF-Access-Client-Id"] == "client-id"
    assert observed["timeout"] == 23


def test_default_openai_factory_uses_owned_tls_client_and_separate_query(monkeypatch):
    import ssl
    import certifi
    import httpx

    clients = []
    def http_client(base_url, *, verify):
        assert isinstance(verify, ssl.SSLContext)
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        clients.append(client)
        return client
    monkeypatch.setattr("agent.process_bootstrap.build_keepalive_http_client", http_client)
    runtime = ResolvedRuntime.from_mapping({
        "provider": "custom", "model": "model", "api_key": "test-key",
        "base_url": "https://route.invalid/v1?version=test", "ssl_ca_cert": certifi.where(),
        "default_query": {"route": "one"}, "args": [],
    })
    first = build_client_bundle(runtime)
    second = build_client_bundle(runtime)
    try:
        assert first.client._client is not second.client._client
        assert first.runtime.base_url == "https://route.invalid/v1"
        assert first.client.default_query == {"version": "test", "route": "one"}
        assert first.runtime["default_query"] == first.client.default_query
        assert "args" not in first.client_kwargs
        first.client.close()
        assert not second.client.is_closed()
    finally:
        first.client.close()
        second.client.close()
