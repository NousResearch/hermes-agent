"""Public lower-domain endpoint queries used by ACP: no live HTTP in tests."""
from __future__ import annotations

import pytest

from models import catalog_endpoint as endpoint
from models.catalog_configured import (
    declared_model_ids, entry_models_discovered, models_config_is_allowlist,
    discovery_enabled,
)


@pytest.fixture(autouse=True)
def reset_query_cache():
    endpoint.reset_endpoint_catalog_cache()
    yield
    endpoint.reset_endpoint_catalog_cache()


def test_declared_models_accept_legacy_shapes_without_sentinels():
    assert declared_model_ids({
        "__discovered_model_catalog__": True, "First": {}, "first": {}, "second": {},
    }) == ["First", "second"]
    assert declared_model_ids([{"name": "one"}, {"id": "two"}, "two"]) == ["one", "two"]
    assert entry_models_discovered({"models": {"__discovered_model_catalog__": True}})
    assert not models_config_is_allowlist({"one": {}}, False)
    assert models_config_is_allowlist(["one"], False)
    assert not models_config_is_allowlist(["one"], True)
    assert not discovery_enabled({"discover_models": "false"})


def test_generic_endpoint_retains_base_then_v1_fallback_and_bearer(monkeypatch):
    calls = []

    def read(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        return {"data": [{"id": "first"}, {"id": "first"}, {"id": "second"}]} if url.endswith("/v1/models") else None

    monkeypatch.setattr(endpoint, "_read_json", read)
    result = endpoint.discover_endpoint_models(
        base_url="https://relay.example", provider="custom", api_key="secret", timeout=1.5,
    )
    assert result == endpoint.EndpointModels(("first", "second"))
    assert [x[0] for x in calls] == [
        "https://relay.example/models", "https://relay.example/v1/models",
    ]
    assert all(x[1]["Authorization"] == "Bearer secret" for x in calls)


def test_anthropic_transport_sends_no_bearer_and_preserves_explicit_headers(monkeypatch):
    seen = []

    def read(url, headers, timeout):
        seen.append((url, dict(headers)))
        return {"data": [{"id": "claude"}]}

    monkeypatch.setattr(endpoint, "_read_json", read)
    result = endpoint.discover_endpoint_models(
        base_url="https://claude.example/v1", provider="custom", api_key="sk-secret",
        api_mode="anthropic_messages", headers={"X-Custom": "scope"},
    )
    assert result.ids == ("claude",)
    assert seen[0][0] == "https://claude.example/v1/models"
    assert seen[0][1]["x-api-key"] == "sk-secret"
    assert seen[0][1]["anthropic-version"] == "2023-06-01"
    assert seen[0][1]["X-Custom"] == "scope"
    assert "Authorization" not in seen[0][1]


def test_native_empty_is_authoritative_and_never_tries_generic(monkeypatch):
    calls = []

    def read(url, headers, timeout):
        calls.append(url)
        return {"models": []}

    monkeypatch.setattr(endpoint, "_read_json", read)
    result = endpoint.discover_endpoint_models(
        base_url="http://127.0.0.1:11434/v1", provider="custom:ollama",
    )
    assert result == endpoint.EndpointModels((), native_ollama=True)
    assert calls == ["http://127.0.0.1:11434/api/tags"]


def test_native_declared_allowlist_skips_probe(monkeypatch):
    monkeypatch.setattr(endpoint, "_read_json", lambda *a, **kw: 1 / 0)
    assert endpoint.discover_endpoint_models(
        base_url="http://127.0.0.1:11434/v1", provider="custom:ollama",
        preserve_native_models=True,
    ) is None


def test_failed_native_probe_falls_back_to_openai_without_losing_ids(monkeypatch):
    calls = []

    def read(url, headers, timeout):
        calls.append(url)
        if url.endswith("/api/tags"):
            return None
        return {"data": [{"id": "proxy-model"}]}

    monkeypatch.setattr(endpoint, "_read_json", read)
    result = endpoint.discover_endpoint_models(
        base_url="http://127.0.0.1:11434/v1", provider="custom",
    )
    assert result == endpoint.EndpointModels(("proxy-model",))
    assert calls == [
        "http://127.0.0.1:11434/api/tags", "http://127.0.0.1:11434/v1/models",
    ]


def test_refuse_non_http_and_redirects():
    assert endpoint.discover_endpoint_models(base_url="file:///tmp/key", provider="custom") is None
    assert endpoint._NoRedirect().redirect_request(
        None, None, 302, "redirect", {}, "https://other.example/models",
    ) is None

def test_repeated_discovery_uses_bounded_cache_and_token_rotation_invalidates(monkeypatch):
    calls = []

    def read(url, headers, timeout):
        calls.append(headers.get("Authorization"))
        return {"data": [{"id": "fast-model"}]}

    monkeypatch.setattr(endpoint, "_read_json", read)
    kwargs = {"base_url": "https://cache.example/v1", "provider": "custom"}
    first = endpoint.discover_endpoint_models(api_key="first-token", **kwargs)
    second = endpoint.discover_endpoint_models(api_key="first-token", **kwargs)
    rotated = endpoint.discover_endpoint_models(api_key="second-token", **kwargs)
    assert first == second == rotated == endpoint.EndpointModels(("fast-model",))
    assert calls == ["Bearer first-token", "Bearer second-token"]
    assert all("token" not in key for key in endpoint._cache)


def test_failed_discovery_is_temporarily_cached(monkeypatch):
    calls = []

    def fail(url, headers, timeout):
        calls.append(url)
        return None

    monkeypatch.setattr(endpoint, "_read_json", fail)
    kwargs = {"base_url": "https://down.example/v1", "provider": "custom"}
    assert endpoint.discover_endpoint_models(**kwargs) is None
    assert endpoint.discover_endpoint_models(**kwargs) is None
    assert len(calls) == 2  # fallback URLs only on the first failed read

def test_explicit_tls_verifier_reaches_transport_and_separates_cache(monkeypatch):
    import ssl

    seen = []

    def read(url, headers, timeout, verifier=None):
        seen.append(verifier)
        return {"data": [{"id": "trusted"}]}

    monkeypatch.setattr(endpoint, "_read_json", read)
    kwargs = {"base_url": "https://internal.example/v1", "provider": "custom"}
    first = endpoint.discover_endpoint_models(tls_verify=False, **kwargs)
    context = ssl.create_default_context()
    second = endpoint.discover_endpoint_models(tls_verify=context, **kwargs)
    assert first == second == endpoint.EndpointModels(("trusted",))
    assert seen[0] is False
    assert seen[1] is context
    assert len(seen) == 2
