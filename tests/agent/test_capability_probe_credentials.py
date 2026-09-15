"""Capability probes materialize credentials without consuming the chat source."""
from unittest.mock import patch

import httpx
import pytest

from agent import auxiliary_client, image_routing, model_metadata
from hermes_cli import models_local


def test_custom_provider_api_key_route_match():
    from hermes_cli.config_providers import get_custom_provider_api_key
    route_key = "probe" + "-route-" + "credential"
    entry = {"name": "llamacpp-keyed", "base_url": "http://127.0.0.1:8107/v1", "api_key": route_key}
    assert get_custom_provider_api_key("http://127.0.0.1:8107/v1", [entry]) == route_key
    assert get_custom_provider_api_key("http://127.0.0.1:9999/v1", [entry]) == ""
    assert get_custom_provider_api_key("http://127.0.0.1:8107/v1", []) == ""


def test_custom_provider_api_key_key_env_hint(monkeypatch):
    from hermes_cli.config_providers import get_custom_provider_api_key
    env_key = "probe" + "-env-" + "credential"
    monkeypatch.setenv("HERMES_TEST_PROBE_KEY", env_key)
    entry = {"name": "llamacpp-env", "base_url": "http://127.0.0.1:8108/v1", "key_env": "HERMES_TEST_PROBE_KEY"}
    assert get_custom_provider_api_key("http://127.0.0.1:8108/v1", [entry]) == env_key


def test_context_length_probe_forwards_custom_provider_key():
    """An empty api_key reaching the custom-endpoint probe is backfilled from the
    route-matching entry — a keyed local server must not be sprayed with 401s (#105379)."""
    route_key = "probe" + "-route-" + "credential"
    entry = {"name": "llamacpp-keyed", "base_url": "http://127.0.0.1:8107/v1", "api_key": route_key}
    seen = {}

    def fake_resolve(model, base_url, api_key=""):
        seen["api_key"] = api_key
        return 8192

    with patch.object(model_metadata, "_resolve_endpoint_context_length", side_effect=fake_resolve):
        ctx = model_metadata.get_model_context_length(
            "fixture-keyed-model", base_url="http://127.0.0.1:8107/v1",
            provider="custom", custom_providers=[entry])
    assert ctx == 8192
    assert seen["api_key"] == route_key


def test_context_length_probe_keeps_empty_key_without_route_match():
    seen = {}

    def fake_resolve(model, base_url, api_key=""):
        seen["api_key"] = api_key
        return 4096

    entry = {"name": "other-route", "base_url": "http://127.0.0.1:9999/v1", "api_key": "probe" + "-other-" + "credential"}
    with patch.object(model_metadata, "_resolve_endpoint_context_length", side_effect=fake_resolve):
        model_metadata.get_model_context_length(
            "fixture-unkeyed-model", base_url="http://127.0.0.1:8107/v1",
            provider="custom", custom_providers=[entry])
    assert seen["api_key"] == ""


def test_explicit_api_key_wins_over_custom_provider_entry():
    explicit_key = "probe" + "-explicit-" + "credential"
    seen = {}

    def fake_resolve(model, base_url, api_key=""):
        seen["api_key"] = api_key
        return 2048

    entry = {"name": "llamacpp-keyed", "base_url": "http://127.0.0.1:8107/v1", "api_key": "probe" + "-route-" + "credential"}
    with patch.object(model_metadata, "_resolve_endpoint_context_length", side_effect=fake_resolve):
        model_metadata.get_model_context_length(
            "fixture-explicit-model", base_url="http://127.0.0.1:8107/v1",
            api_key=explicit_key, provider="custom", custom_providers=[entry])
    assert seen["api_key"] == explicit_key


@pytest.mark.parametrize("credential,expected", [(lambda: "minted", "minted"), ("static", "static")])
def test_capability_paths_share_concrete_bearer(credential, expected):
    auxiliary_client.set_runtime_main("custom", "fixture", api_key=credential)
    try:
        assert image_routing._resolve_inference_api_key({}, "custom") == expected
        assert model_metadata._auth_headers(credential) == {"Authorization": f"Bearer {expected}"}
        assert models_local._lmstudio_request_headers(credential)["Authorization"] == f"Bearer {expected}"
        requests = []
        def capture(req):
            requests.append(req)
            return httpx.Response(200, json={"capabilities": ["thinking"]})
        client_type = httpx.Client
        with patch("httpx.Client", lambda **kwargs: client_type(
            **kwargs, transport=httpx.MockTransport(capture)
        )):
            models_local.ollama_model_supports_thinking("fixture", "http://localhost:11434/v1", credential)
        assert requests and requests[0].headers["Authorization"] == f"Bearer {expected}"
        assert auxiliary_client._runtime_main_value("api_key") is credential
    finally:
        auxiliary_client.clear_runtime_main()


def test_failed_callable_never_becomes_a_bearer(monkeypatch):
    def failed():
        raise RuntimeError("secret-bearing command failure")
    for value in (failed, lambda: object(), object(), None):
        assert model_metadata._auth_headers(value) == {}
        assert "Authorization" not in models_local._lmstudio_request_headers(value)

    from hermes_cli import models
    url = "http://localhost:11434/v1"
    configured = {"base_url": url, "api_key": "provider-fallback", "extra_headers": {
        "aUtHoRiZaTiOn": "Bearer configured-fallback", "X-Probe-Fixture": "preserved",
    }}
    monkeypatch.setattr(models, "_get_provider_config_dict", lambda _: configured)
    for value in (failed, lambda: object(), lambda: ""):
        auxiliary_client.set_runtime_main("custom", "fixture", api_key=value)
        try:
            for cfg in ({"model": {"api_key": "model-fallback"}},
                        {"providers": {"custom": {"api_key": "provider-fallback"}}}):
                assert image_routing._resolve_inference_api_key(cfg, "custom") == ""
            assert models_local._get_ollama_native_headers(url, api_key=value) == {
                "X-Probe-Fixture": "preserved",
            }
            assert auxiliary_client._runtime_main_value("api_key") is value
        finally:
            auxiliary_client.clear_runtime_main()
    # An absent explicit credential still permits configured authentication.
    assert models_local._get_ollama_native_headers(url)["aUtHoRiZaTiOn"] == "Bearer configured-fallback"
    assert image_routing._resolve_inference_api_key({"model": {"api_key": "model-fallback"}}, "custom") == "model-fallback"
