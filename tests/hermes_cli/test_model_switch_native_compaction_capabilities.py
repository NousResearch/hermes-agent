"""Regression coverage for trusted native-compaction routes selected by /model."""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


def test_switch_preserves_named_provider_capabilities_with_shared_base_url():
    """The resolver identity, not the first same-URL config entry, owns capability policy."""
    base_url = "https://proxy.example.test/v1"
    custom_providers = [
        {
            "name": "generic",
            "base_url": base_url,
            "models": ["other-model"],
            "capabilities": {},
        },
        {
            "name": "chatgpt-tier",
            "base_url": base_url,
            "models": ["gpt-5.6-sol"],
            "capabilities": {"openai_native_compaction": True},
        },
    ]
    accepted = {"accepted": True, "persist": True, "recognized": True, "message": ""}

    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.model_switch.normalize_model_for_provider",
            side_effect=lambda model, _provider: model,
        ),
        patch("hermes_cli.models_validate.validate_requested_model", return_value=accepted),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": base_url,
                "api_mode": "codex_responses",
                "capabilities": {"openai_native_compaction": True},
            },
        ),
    ):
        result = switch_model(
            raw_input="gpt-5.6-sol",
            current_provider="custom",
            current_model="gpt-5.6-terra",
            current_base_url=base_url,
            custom_providers=custom_providers,
        )

    assert result.success is True, result.error_message
    assert result.provider_capabilities == {"openai_native_compaction": True}
    assert result.runtime_capabilities == {"native_compaction": True}