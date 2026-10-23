"""Capability probes materialize credentials without consuming the chat source."""
from unittest.mock import patch

import httpx
import pytest

from agent import auxiliary_client, image_routing, model_metadata
from hermes_cli import models_local


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
