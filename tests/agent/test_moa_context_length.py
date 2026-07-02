from __future__ import annotations

from unittest.mock import patch


def test_moa_context_length_uses_aggregator_custom_provider_override():
    """MoA should resolve its acting aggregator's configured context length.

    The virtual ``moa`` provider's model is just a preset name.  The acting
    context window belongs to the preset's aggregator, including per-model
    custom-provider overrides threaded in by the caller.
    """
    from agent.model_metadata import get_model_context_length

    custom_providers = [
        {
            "name": "private-router",
            "base_url": "https://private.example/v1",
            "models": {
                "private-32k": {
                    "context_length": 32_768,
                },
            },
        }
    ]

    with patch("hermes_cli.config.load_config", return_value={"moa": {}}), patch(
        "hermes_cli.moa_config.resolve_moa_preset",
        return_value={
            "aggregator": {"provider": "private-router", "model": "private-32k"},
            "reference_models": [],
        },
    ), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value={
            "provider": "private-router",
            "model": "private-32k",
            "base_url": "https://private.example/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
        },
    ):
        assert (
            get_model_context_length(
                "default",
                provider="moa",
                custom_providers=custom_providers,
            )
            == 32_768
        )
