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


def test_failed_callable_never_becomes_a_bearer():
    def failed():
        raise RuntimeError("secret-bearing command failure")
    for value in (failed, lambda: object(), object(), None):
        assert model_metadata._auth_headers(value) == {}
        assert "Authorization" not in models_local._lmstudio_request_headers(value)
