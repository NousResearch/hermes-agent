"""Regression tests for OpenRouter auth validation during `/model` switching."""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_explicit_openrouter_switch_requires_openrouter_api_key(monkeypatch):
    """Refuse `/model --provider openrouter` when only OPENAI_API_KEY exists."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")

    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-openai-fallback",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            },
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        result = switch_model(
            raw_input="openai/gpt-5.4",
            current_provider="nous",
            current_model="hermes-3-llama-3.1-8b",
            explicit_provider="openrouter",
        )

    assert result.success is False
    assert "OPENROUTER_API_KEY" in result.error_message
    assert "HTTP 400" in result.error_message


def test_explicit_openrouter_switch_accepts_real_openrouter_key(monkeypatch):
    """Allow the switch once OPENROUTER_API_KEY is actually configured."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")

    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-or-real",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            },
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        result = switch_model(
            raw_input="openai/gpt-5.4",
            current_provider="nous",
            current_model="hermes-3-llama-3.1-8b",
            explicit_provider="openrouter",
        )

    assert result.success is True
    assert result.target_provider == "openrouter"
    assert result.api_key == "sk-or-real"
