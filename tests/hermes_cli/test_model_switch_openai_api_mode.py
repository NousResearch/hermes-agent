"""Regression tests for OpenAI-direct api_mode recomputation during /model switch.

api.openai.com only accepts the Responses API (codex_responses) for its
reasoning models when tools + reasoning are in play — /v1/chat/completions
returns HTTP 400 ("Function tools with reasoning_effort are not supported").

When a session switches to a GPT-5.x model on api.openai.com while carrying a
stale ``chat_completions`` api_mode from the previous provider (e.g. an
openrouter default), the switch must override the stale value with the
host-mandated ``codex_responses``. Filling only when empty (the earlier fix)
was insufficient: the carried value was a *non-empty but wrong* mode.

This surfaced after the reasoning-unification refactor made the switched
model's reasoning effort actually apply, turning a silently-dropped
reasoning_effort into a live one that OpenAI 400s on the chat_completions path.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model

_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _run_openai_switch(
    raw_input: str,
    current_provider: str = "openrouter",
    current_model: str = "anthropic/claude-opus-4.8",
    explicit_provider: str = "openai-api",
    runtime_api_mode: str = "chat_completions",
    runtime_base_url: str = "https://api.openai.com/v1",
):
    """Run switch_model with OpenAI-direct mocks and return the result."""
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-test",
                "base_url": runtime_base_url,
                "api_mode": runtime_api_mode,
            },
        ),
        patch(
            "hermes_cli.models.validate_requested_model",
            return_value=_MOCK_VALIDATION,
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            explicit_provider=explicit_provider,
        )


def test_stale_chat_completions_overridden_on_openai_direct():
    """The incident: stale chat_completions on api.openai.com → codex_responses.

    Switching from an openrouter/chat_completions session to gpt-5.6-sol on
    api.openai.com must flip the wire protocol to the Responses API so tools +
    reasoning are preserved, instead of 400ing on chat/completions.
    """
    result = _run_openai_switch(
        raw_input="gpt-5.6-sol",
        current_provider="openrouter",
        current_model="anthropic/claude-opus-4.8",
        explicit_provider="openai-api",
        runtime_api_mode="chat_completions",  # stale value carried over
    )

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.target_provider == "openai-api"
    assert result.new_model == "gpt-5.6-sol"
    assert result.api_mode == "codex_responses"


def test_empty_api_mode_filled_on_openai_direct():
    """An empty runtime api_mode on api.openai.com resolves to codex_responses."""
    result = _run_openai_switch(
        raw_input="gpt-5.6-sol",
        runtime_api_mode="",  # empty — the earlier fill-if-empty subcase
    )

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.api_mode == "codex_responses"


def test_generic_endpoint_keeps_explicit_api_mode():
    """A generic (non-host-mandated) endpoint must NOT have its api_mode clobbered.

    Only hosts that mandate one wire protocol override a carried value; a
    generic OpenAI-compatible relay returns None from host_mandated_api_mode,
    so the switch path leaves the resolver's api_mode untouched.
    """
    from hermes_cli.providers import host_mandated_api_mode

    assert host_mandated_api_mode("https://generic.example.com/v1") is None
    # Lookalike / path-spoof hosts must also NOT be treated as mandated (#32243).
    assert host_mandated_api_mode("https://api.openai.com.attacker.test/v1") is None
    assert host_mandated_api_mode("https://proxy.test/api.openai.com/v1") is None
    # The real endpoints are mandated.
    assert host_mandated_api_mode("https://api.openai.com/v1") == "codex_responses"
    assert host_mandated_api_mode("https://api.anthropic.com") == "anthropic_messages"
