"""Regression tests for xAI curated + models.dev picker-time merge."""

from unittest.mock import patch

from hermes_cli.models import (
    _PROVIDER_MODELS,
    provider_model_ids,
)






def test_xai_oauth_picker_merges_models_dev_at_call_time():
    """xai-oauth must not return the import-frozen list; merge at picker time."""
    top = _PROVIDER_MODELS["xai-oauth"][0]
    mdev = ["grok-build-0.1", "grok-new-from-models-dev", top]
    with patch("agent.models_dev.list_agentic_models", return_value=mdev) as mocked:
        models = provider_model_ids("xai-oauth")

    mocked.assert_called()
    assert models[0] == top
    assert "grok-new-from-models-dev" in models


def test_xai_api_key_picker_merges_models_dev_when_live_unavailable():
    """Without a live /v1/models hit, xai uses the models.dev preferred path."""
    top = _PROVIDER_MODELS["xai-oauth"][0]
    mdev = ["grok-build-0.1", "grok-new-from-models-dev", top]
    with (
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            side_effect=Exception("no key"),
        ),
        patch("agent.models_dev.list_agentic_models", return_value=mdev) as mocked,
    ):
        models = provider_model_ids("xai")

    mocked.assert_called()
    assert "grok-new-from-models-dev" in models
    assert models[0] == top


def test_xai_pin_survives_when_top_model_only_in_extras():
    """If models.dev omits the curated top model, curated extras + finalize still pin it."""
    top = _PROVIDER_MODELS["xai-oauth"][0]
    mdev = ["grok-build-0.1", "grok-new-from-models-dev"]
    with patch("agent.models_dev.list_agentic_models", return_value=mdev):
        models = provider_model_ids("xai-oauth")

    assert models[0] == top
