"""Unit tests for the Custom/Ollama provider profile's thinking-mode wiring.

When ``reasoning_effort: none`` (or ``enabled: False``) is set, Hermes must
suppress Qwen3/DeepSeek-style thinking on Ollama.  Ollama exposes two knobs:

  1. ``extra_body["think"] = False``  — honoured on the native /api/chat endpoint.
  2. top-level ``reasoning_effort = "none"`` — honoured on /v1/chat/completions
     (the OpenAI-compat path that Hermes actually uses).

The original code only emitted (1), which Ollama silently ignores on /v1/chat/completions,
causing every request to run the full reasoning chain regardless of the user's config
(NousResearch/hermes-agent#6152, #25758).  Both fields must be emitted together.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom/Ollama provider profile."""
    import model_tools  # noqa: F401 — triggers plugin discovery
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomProfileThinkingDisabled:
    """When reasoning is disabled, both think=False and reasoning_effort='none' are emitted."""

    def test_effort_none_emits_both_fields(self, custom_profile):
        """``reasoning_effort: none`` → extra_body.think=False + top-level reasoning_effort='none'."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
        )
        assert extra_body.get("think") is False, "extra_body.think must be False for /api/chat compat"
        assert top_level.get("reasoning_effort") == "none", (
            "top-level reasoning_effort must be 'none' for /v1/chat/completions"
        )

    def test_enabled_false_emits_both_fields(self, custom_profile):
        """``enabled: False`` → same wire shape as effort=none."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
        )
        assert extra_body.get("think") is False
        assert top_level.get("reasoning_effort") == "none"

    def test_enabled_false_with_any_effort_emits_both_fields(self, custom_profile):
        """``enabled: False`` takes precedence over any effort value."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
        )
        assert extra_body.get("think") is False
        assert top_level.get("reasoning_effort") == "none"


class TestCustomProfileThinkingEnabled:
    """When reasoning is active (default), neither suppression field is emitted."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """No reasoning_config → no think field, no reasoning_effort."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config=None,
        )
        assert "think" not in extra_body
        assert "reasoning_effort" not in top_level

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", ""])
    def test_active_efforts_do_not_suppress(self, custom_profile, effort):
        """Efforts other than 'none' must not suppress thinking."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
        )
        assert "think" not in extra_body
        assert "reasoning_effort" not in top_level

    def test_enabled_true_medium_effort_emits_nothing(self, custom_profile):
        """Default config (enabled=True, effort=medium) → no suppression fields."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "think" not in extra_body
        assert "reasoning_effort" not in top_level


class TestCustomProfileOllamaNumCtx:
    """ollama_num_ctx is forwarded into extra_body.options regardless of reasoning config."""

    def test_num_ctx_forwarded(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config=None,
            ollama_num_ctx=65536,
        )
        assert extra_body.get("options", {}).get("num_ctx") == 65536

    def test_num_ctx_plus_disabled_thinking(self, custom_profile):
        """num_ctx and think=False can coexist in extra_body."""
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            ollama_num_ctx=131072,
        )
        assert extra_body.get("options", {}).get("num_ctx") == 131072
        assert extra_body.get("think") is False
        assert top_level.get("reasoning_effort") == "none"

    def test_no_num_ctx_means_no_options_key(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config=None,
            ollama_num_ctx=None,
        )
        assert "options" not in extra_body


class TestCustomProfileFullKwargsIntegration:
    """End-to-end: ChatCompletionsTransport produces the correct wire shape."""

    def test_thinking_disabled_full_kwargs(self, custom_profile):
        """reasoning_effort=none → top-level + extra_body both carry suppression."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="qwen3.6:35b",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=custom_profile,
            reasoning_config={"enabled": True, "effort": "none"},
            base_url="http://172.18.0.1:11434/v1",
            provider_name="custom",
        )
        # Top-level field — what Ollama /v1/chat/completions actually honours
        assert kwargs.get("reasoning_effort") == "none", (
            "reasoning_effort must be at top-level for Ollama /v1/chat/completions"
        )
        # extra_body field — backward-compat for /api/chat and proxies
        assert kwargs.get("extra_body", {}).get("think") is False, (
            "extra_body.think=False must be present for /api/chat compat"
        )

    def test_thinking_enabled_full_kwargs(self, custom_profile):
        """Default config → no suppression fields on the wire."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="qwen3.6:35b",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=custom_profile,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="http://172.18.0.1:11434/v1",
            provider_name="custom",
        )
        assert "reasoning_effort" not in kwargs
        assert "think" not in kwargs.get("extra_body", {})
