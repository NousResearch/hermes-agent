"""`/model <alias>` stays on an external-process provider that declares the alias.

Process providers are not in models.dev, so ``list_provider_models`` returns ``[]`` and
the alias fallback used to route ``haiku`` to whichever authenticated HTTP provider was
handy (live repro: Copilot, then HTTP 400 on the next turn).
"""
from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from providers.base import ProviderProfile

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _switch(raw_input, profile):
    with patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("providers.get_provider_profile", return_value=profile), \
         patch("hermes_cli.model_switch.get_authenticated_provider_slugs", return_value=["copilot", "anthropic"]), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=("copilot", "claude-haiku-4.5")), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "external-process", "base_url": profile.base_url, "api_mode": "chat_completions"}):
        return switch_model(raw_input=raw_input, current_provider=profile.name,
                            current_model="claude-sonnet-5[1m]", current_base_url=profile.base_url,
                            user_providers={}, custom_providers=[])


def test_alias_and_model_id_stay_on_external_process_provider():
    profile = ProviderProfile(
        name="proc-provider", auth_type="external_process", base_url="process://proc-provider",
        fallback_models=("claude-sonnet-5[1m]", "claude-haiku-4-5-20251001", "claude-opus-5[1m]", "claude-opus-4-8[1m]"),
        model_aliases={"opus": "claude-opus-5[1m]", "fable": "claude-fable-5-1[1m]"})
    for typed, expected in (("haiku", "claude-haiku-4-5-20251001"), ("claude-opus-5[1m]", "claude-opus-5[1m]"),
                            ("Claude-Opus-5", "claude-opus-5[1m]"), ("opus", "claude-opus-5[1m]"),
                            ("fable", "claude-fable-5-1[1m]")):
        result = _switch(typed, profile)
        assert result.success, result.error_message
        assert (result.target_provider, result.new_model) == (profile.name, expected), typed


def test_declared_process_model_validates_without_endpoint_warning():
    from hermes_cli.models_validate import validate_requested_model
    profile = ProviderProfile(
        name="proc-provider", auth_type="external_process", base_url="process://proc-provider",
        fallback_models=("claude-sonnet-5[1m]", "claude-haiku-4-5-20251001"))
    with patch("providers.get_provider_profile", return_value=profile), \
         patch("hermes_cli.models.fetch_api_models", side_effect=AssertionError("no HTTP probe for process://")):
        verdict = validate_requested_model("claude-haiku-4-5-20251001", profile.name, base_url=profile.base_url)
        assert verdict["accepted"] and verdict["recognized"] and not verdict.get("message")
        unknown = validate_requested_model("claude-haiku-9", profile.name, base_url=profile.base_url)
        assert "unreachable" not in (unknown.get("message") or "")
