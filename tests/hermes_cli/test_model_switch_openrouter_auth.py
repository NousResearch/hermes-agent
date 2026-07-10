"""Regression tests for OpenRouter auth validation during ``/model``."""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_ACCEPTED = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_auto_detected_openrouter_switch_fails_before_persisting_without_credentials(
    tmp_path, monkeypatch
):
    """Exercise real model detection and runtime auth in an isolated home."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    result = switch_model(
        raw_input="anthropic/claude-fable-5",
        current_provider="openai-codex",
        current_model="gpt-5.6-sol",
    )

    assert result.success is False
    assert result.target_provider == "openrouter"
    assert "OpenRouter credentials" in result.error_message
    assert "--provider anthropic" in result.error_message


def test_auto_detected_openrouter_switch_still_accepts_resolved_credentials():
    """Keep intentional OpenRouter setups working regardless of key source."""
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.models.detect_provider_for_model",
            return_value=("openrouter", "anthropic/claude-fable-5"),
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openrouter",
                "api_key": "resolved-openrouter-credential",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            },
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=_ACCEPTED),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
    ):
        result = switch_model(
            raw_input="anthropic/claude-fable-5",
            current_provider="openai-codex",
            current_model="gpt-5.6-sol",
        )

    assert result.success is True
    assert result.target_provider == "openrouter"
    assert result.api_key == "resolved-openrouter-credential"


def test_openai_slug_suggests_authenticated_direct_codex_route():
    """Legacy ``openai`` aliasing must not hide an available direct route."""
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.models.detect_provider_for_model",
            return_value=("openrouter", "openai/gpt-5.6-sol"),
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openrouter",
                "api_key": "",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            },
        ),
        patch(
            "hermes_cli.model_switch.get_authenticated_provider_slugs",
            return_value=["openai-codex"],
        ),
    ):
        result = switch_model(
            raw_input="openai/gpt-5.6-sol",
            current_provider="openai-codex",
            current_model="gpt-5.6-terra",
        )

    assert result.success is False
    assert "--provider openai-codex" in result.error_message
