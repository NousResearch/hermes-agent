"""Regression tests for xAI curated + models.dev picker-time merge."""

from unittest.mock import patch

from hermes_cli.models import (
    _MODELS_DEV_PREFERRED,
    _PROVIDER_MODELS,
    provider_model_ids,
)


def test_grok_4_6_is_default_pin():
    models = _PROVIDER_MODELS["xai-oauth"]
    assert models[0] == "grok-4.6"


def test_xai_providers_are_models_dev_preferred():
    assert "xai" in _MODELS_DEV_PREFERRED
    assert "xai-oauth" in _MODELS_DEV_PREFERRED


def test_xai_oauth_picker_merges_models_dev_at_call_time():
    """xai-oauth must not return the import-frozen list; merge at picker time."""
    mdev = ["grok-build-0.1", "grok-new-from-models-dev", "grok-4.6"]
    with patch("agent.models_dev.list_agentic_models", return_value=mdev) as mocked:
        models = provider_model_ids("xai-oauth")

    mocked.assert_called()
    assert models[0] == "grok-4.6"
    assert "grok-new-from-models-dev" in models


def test_xai_api_key_picker_merges_models_dev_when_live_unavailable():
    """Without a live /v1/models hit, xai uses the models.dev preferred path."""
    mdev = ["grok-build-0.1", "grok-new-from-models-dev", "grok-4.6"]
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
    assert models[0] == "grok-4.6"


def test_xai_pin_survives_when_top_model_only_in_extras():
    """If models.dev omits grok-4.6, curated extras + finalize still pin it."""
    mdev = ["grok-build-0.1", "grok-new-from-models-dev"]
    with patch("agent.models_dev.list_agentic_models", return_value=mdev):
        models = provider_model_ids("xai-oauth")

    assert models[0] == "grok-4.6"
