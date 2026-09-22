"""``/model`` within one named custom provider recomputes ``api_mode`` from
``providers.<id>.models.<model>.transport``.

Mirrors ``test_model_switch_copilot_api_mode.py``: the runtime's (possibly stale) ``api_mode``
must be overridden by the wire the NEW model declares, and a model declaring nothing falls back
to the entry-level transport rather than inheriting the previous model's wire.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_BASE = "https://gateway.example.com/v1"

_MOCK_VALIDATION = {"accepted": True, "persist": True, "recognized": True, "message": None}

_MODELS = {
    "claude-sonnet-5": {"transport": "anthropic_messages"},
    "gpt-5.4-mini": {"transport": "codex_responses"},
    "glm-5.3": {},
}
# ``providers:`` dict as the CLI/gateway hand it over, plus the compat ``custom_providers`` view
# the same callers pass alongside (the explicit-provider route is resolved from that view).
_USER_PROVIDERS = {
    "gateway": {"api": _BASE, "key_env": "GATEWAY_API_KEY", "transport": "chat_completions",
                "default_model": "glm-5.3", "models": _MODELS},
}
_CUSTOM_PROVIDERS = [
    {"name": "gateway", "provider_key": "gateway", "base_url": _BASE, "key_env": "GATEWAY_API_KEY",
     "api_mode": "chat_completions", "model": "glm-5.3", "models": _MODELS},
]


def _switch(raw_input: str, runtime_api_mode: str, current_model: str = "glm-5.3"):
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"api_key": "test-key", "base_url": _BASE, "api_mode": runtime_api_mode},
        ),
        patch("hermes_cli.models_validate.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider="custom:gateway",
            current_model=current_model,
            explicit_provider="custom:gateway",
            user_providers=_USER_PROVIDERS,
            custom_providers=_CUSTOM_PROVIDERS,
        )


def test_switch_to_messages_model_flips_stale_chat_api_mode():
    result = _switch("claude-sonnet-5", runtime_api_mode="chat_completions")
    assert result.success, result.error_message
    assert result.new_model == "claude-sonnet-5"
    assert result.target_provider == "custom:gateway"
    assert result.api_mode == "anthropic_messages"


def test_switch_to_responses_model_flips_stale_messages_api_mode():
    result = _switch("gpt-5.4-mini", runtime_api_mode="anthropic_messages", current_model="claude-sonnet-5")
    assert result.success, result.error_message
    assert result.api_mode == "codex_responses"


def test_switch_to_undeclared_model_returns_to_entry_transport():
    """A model with no per-model transport must not inherit the previous model's wire."""
    result = _switch("glm-5.3", runtime_api_mode="codex_responses", current_model="gpt-5.4-mini")
    assert result.success, result.error_message
    assert result.api_mode == "chat_completions"


def test_configured_lookup_reads_either_config_shape():
    from hermes_cli.model_switch import _configured_model_api_mode
    assert _configured_model_api_mode("claude-sonnet-5", _BASE, _USER_PROVIDERS, None) == "anthropic_messages"
    assert _configured_model_api_mode("claude-sonnet-5", _BASE, None, _CUSTOM_PROVIDERS) == "anthropic_messages"
    # Undeclared model → the entry-level transport; unknown route → nothing.
    assert _configured_model_api_mode("glm-5.3", _BASE, _USER_PROVIDERS, None) == "chat_completions"
    assert _configured_model_api_mode("claude-sonnet-5", "https://elsewhere.example.com/v1", _USER_PROVIDERS, None) == ""


def test_host_mandated_wire_beats_per_model_transport():
    """api.anthropic.com accepts exactly one wire; a per-model ``codex_responses`` there is ignored."""
    from hermes_cli.model_switch import model_derived_api_mode
    providers = {"direct": {"api": "https://api.anthropic.com", "key_env": "K",
                            "models": {"claude-sonnet-5": {"transport": "codex_responses"}}}}
    assert model_derived_api_mode("custom:direct", "claude-sonnet-5", base_url="https://api.anthropic.com",
                                  user_providers=providers) is None
    assert model_derived_api_mode("custom:gateway", "claude-sonnet-5", base_url=_BASE,
                                  user_providers=_USER_PROVIDERS) == "anthropic_messages"
    # Built-in providers keep their own table; nothing is derived for a plain vendor.
    assert model_derived_api_mode("openrouter", "claude-sonnet-5", base_url="https://openrouter.ai/api/v1",
                                  user_providers=_USER_PROVIDERS) is None
