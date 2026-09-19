"""A family alias missing on a single-provider setup must not silently leave it (#114475, #114477).

With ``provider: bedrock``, ``/model opus`` resolved through the built-in alias table onto the
user's *other* authenticated providers (OpenRouter) and reported success. Crossing providers
unasked is only legitimate on a routing aggregator; elsewhere the switch must fail on the
configured provider with a mismatch message.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run_switch(*, raw_input, current_provider, resolve_alias_side_effect):
    with patch(
        "hermes_cli.model_switch.resolve_alias", side_effect=resolve_alias_side_effect
    ), patch(
        "hermes_cli.model_switch.get_authenticated_provider_slugs",
        return_value=["openrouter", "anthropic"],
    ), patch(
        "hermes_cli.providers.is_routing_aggregator",
        side_effect=lambda slug: slug == "openrouter",
    ), patch(
        "hermes_cli.model_switch.list_provider_models", return_value=[]
    ), patch(
        "hermes_cli.model_switch.normalize_model_for_provider",
        side_effect=lambda model, provider: model,
    ), patch(
        "hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED
    ), patch(
        "hermes_cli.models.detect_provider_for_model", return_value=None
    ), patch(
        "hermes_cli.model_switch.get_model_info", return_value=None
    ), patch(
        "hermes_cli.model_switch.get_model_capabilities", return_value=None
    ), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value={"api_key": "***", "base_url": "http://resolved/v1", "api_mode": ""},
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="old-model",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        )


def _miss_then_cross(raw, provider):
    """Step-a miss everywhere; the fallback finds opus on anthropic."""
    if provider == "anthropic":
        return ("anthropic", "claude-opus-4-6", "opus")
    return None


def test_native_provider_alias_missing_fails_without_crossing():
    result = _run_switch(
        raw_input="opus",
        current_provider="bedrock",
        resolve_alias_side_effect=_miss_then_cross,
    )
    assert result.success is False, result
    assert "bedrock" in (result.error_message or "")
    assert result.target_provider != "openrouter"
    assert result.target_provider != "anthropic"


def test_routing_aggregator_alias_fallback_still_crosses():
    result = _run_switch(
        raw_input="opus",
        current_provider="openrouter",
        resolve_alias_side_effect=_miss_then_cross,
    )
    assert result.success is True, getattr(result, "error_message", result)
    assert result.target_provider == "anthropic"
    assert result.new_model == "claude-opus-4-6"
