"""Regression tests for live model discovery in the /model picker."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


@patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}, clear=True)
def test_api_key_provider_picker_uses_live_models_when_available():
    live_models = [
        "nvidia/nemotron-ultra-253b-v1",
        "meta/llama-4-maverick-17b-128e-instruct",
    ]

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models.provider_model_ids", return_value=live_models) as provider_ids:
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    nvidia = next((p for p in providers if p["slug"] == "nvidia"), None)

    assert nvidia is not None
    assert nvidia["models"] == live_models
    assert nvidia["total_models"] == len(live_models)
    provider_ids.assert_any_call("nvidia")
