"""Behavior regressions for NVIDIA's offline model-picker catalog."""

from unittest.mock import patch

from providers import get_provider_profile

from hermes_cli.models import (
    _MODELS_DEV_PREFERRED,
    _PROVIDER_MODELS,
    provider_model_ids,
)


def test_nvidia_offline_catalog_uses_curated_models_before_profile_fallback():
    """NVIDIA's small profile fallback must not hide its curated catalog."""
    profile = get_provider_profile("nvidia")
    assert profile is not None
    assert "nvidia" in _MODELS_DEV_PREFERRED
    assert len(profile.fallback_models) < len(_PROVIDER_MODELS["nvidia"])
    assert list(profile.fallback_models) != _PROVIDER_MODELS["nvidia"]

    with (
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "", "base_url": ""},
        ),
        patch("agent.models_dev.list_agentic_models", return_value=[]),
    ):
        models = provider_model_ids("nvidia")

    assert models == _PROVIDER_MODELS["nvidia"]


def test_nvidia_unavailable_live_catalog_merges_fresh_models_dev_entries():
    """An empty NVIDIA API response still uses the models.dev-preferred path."""
    profile = get_provider_profile("nvidia")
    assert profile is not None
    fresh_model = "nvidia/new-model-from-models-dev"

    with (
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "api_key": "nvapi-test",
                "base_url": "https://integrate.api.nvidia.com/v1",
            },
        ),
        patch.object(profile, "fetch_models", return_value=[]) as fetch_models,
        patch("agent.models_dev.list_agentic_models", return_value=[fresh_model]),
    ):
        models = provider_model_ids("nvidia")

    fetch_models.assert_called_once()
    assert models[0] == fresh_model
    assert set(_PROVIDER_MODELS["nvidia"]) <= set(models)
