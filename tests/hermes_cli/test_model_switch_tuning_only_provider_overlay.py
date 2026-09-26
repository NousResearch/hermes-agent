"""A ``providers.<built-in>`` block that only carries tuning keys (``request_timeout_seconds``,
``stale_timeout_seconds``, ``models: {}``) overlays the built-in provider; it declares no endpoint
of its own. ``/model --provider openai-codex gpt-x`` must validate against the built-in
``openai-codex`` branch (curated catalog + hidden-slug soft-accept), not as a ``custom:`` endpoint.

Regression: with such a block on disk, every Codex switch from the Desktop onboarding / ``/model``
was rejected with "could not reach this custom endpoint's model listing at
``chatgpt.com/backend-api/codex/models`` … was not saved" because the custom-endpoint probe (a
generic Bearer ``GET /models``) is the wrong shape for the Codex backend. The credential path already
treated the block as a built-in overlay (no ``base_url`` → built-in resolver); validation must
follow the same rule.

Hermetic: credential resolution and the validator are patched at the seams ``switch_model`` reads,
and the validator records which provider slug it was asked to validate against.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _switch(explicit_provider, user_providers, raw_input="gpt-6-astra"):
    seen = {}

    def _validate(model_name, provider, **kwargs):
        seen["provider"] = provider
        seen["base_url"] = kwargs.get("base_url")
        return _ACCEPTED

    with patch("hermes_cli.models_validate.validate_requested_model", side_effect=_validate), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={"api_key": "***", "base_url": "https://chatgpt.com/backend-api/codex", "api_mode": ""},
         ):
        result = switch_model(
            raw_input=raw_input, explicit_provider=explicit_provider,
            current_provider="anthropic", current_model="claude-sonnet-4-6",
            current_base_url="https://api.anthropic.com", current_api_key="",
            user_providers=user_providers, custom_providers=[],
        )
    return result, seen


def test_tuning_only_builtin_overlay_validates_as_builtin_provider():
    user_providers = {
        "openai-codex": {"models": "{}", "request_timeout_seconds": 1800, "stale_timeout_seconds": 300},
    }
    result, seen = _switch("openai-codex", user_providers)
    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"
    assert seen["provider"] == "openai-codex"  # not ``custom:openai-codex``
    assert not str(seen["provider"]).startswith("custom")


def test_user_block_with_own_endpoint_still_validates_as_custom():
    """Negative invariant: a ``providers.<key>`` block that DOES declare an endpoint keeps the
    custom-endpoint validation (an unlisted id is soft-accepted there, never hard-rejected)."""
    user_providers = {
        "hyper": {"base_url": "http://127.0.0.1:9/v1", "api_key_env": "HYPER_KEY"},
    }
    with patch.dict("os.environ", {"HYPER_KEY": "test-key"}):
        result, seen = _switch("hyper", user_providers, raw_input="deepseek-v4.1-flash")
    assert result.success is True, result.error_message
    assert seen["provider"] == "custom:hyper"
