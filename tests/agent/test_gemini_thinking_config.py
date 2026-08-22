"""Tests for Gemini thinking config — reasoning disabled sets thinkingBudget: 0.

Issue #91927: Gemini models bill thought tokens against maxOutputTokens even
when ``includeThoughts: False`` is set. To truly disable thinking so thought
tokens don't starve a small max_tokens budget (e.g. title generation's 64
tokens), ``thinkingBudget: 0`` must be set on models that support it.
"""

from agent.transports.chat_completions import (
    _build_gemini_thinking_config,
    _snake_case_gemini_thinking_config,
)


class TestBuildGeminiThinkingConfigDisabled:
    """When reasoning is disabled, thinkingBudget must be 0 on supported models."""

    def test_disabled_sets_thinking_budget_zero_gemini_25(self):
        """Gemini 2.5 supports thinkingBudget; disabling reasoning must set it to 0."""
        config = _build_gemini_thinking_config("gemini-2.5-flash", {"enabled": False})
        assert config is not None
        assert config.get("includeThoughts") is False
        assert config.get("thinkingBudget") == 0

    def test_disabled_sets_thinking_budget_zero_gemini_3(self):
        """Gemini 3.x supports thinkingBudget; disabling reasoning must set it to 0."""
        config = _build_gemini_thinking_config("gemini-3.6-flash", {"enabled": False})
        assert config is not None
        assert config.get("includeThoughts") is False
        assert config.get("thinkingBudget") == 0

    def test_disabled_sets_thinking_budget_zero_gemini_3_pro(self):
        config = _build_gemini_thinking_config("gemini-3.1-pro", {"enabled": False})
        assert config is not None
        assert config.get("includeThoughts") is False
        assert config.get("thinkingBudget") == 0

    def test_disabled_no_thinking_budget_on_older_gemini(self):
        """Older Gemini models (pre-2.5) don't support thinkingBudget; only
        includeThoughts: False should be set."""
        config = _build_gemini_thinking_config("gemini-1.5-flash", {"enabled": False})
        assert config is not None
        assert config.get("includeThoughts") is False
        assert "thinkingBudget" not in config

    def test_disabled_no_thinking_budget_on_non_gemini(self):
        """Non-Gemini models must not get a thinking config at all."""
        config = _build_gemini_thinking_config("gpt-4o", {"enabled": False})
        assert config is None

    def test_disabled_no_thinking_budget_on_gemma(self):
        """Gemma models use the gemini provider but reject thinking_config."""
        config = _build_gemini_thinking_config("gemma-2b", {"enabled": False})
        assert config is None

    def test_effort_none_also_sets_thinking_budget_zero(self):
        """effort='none' is equivalent to enabled=False and must also set
        thinkingBudget: 0 on supported models."""
        config = _build_gemini_thinking_config("gemini-2.5-flash", {"effort": "none"})
        assert config is not None
        assert config.get("includeThoughts") is False
        assert config.get("thinkingBudget") == 0


class TestSnakeCaseGeminiThinkingConfig:
    """Verify thinkingBudget is translated to thinking_budget for OpenAI-compat."""

    def test_thinking_budget_translated(self):
        config = {"includeThoughts": False, "thinkingBudget": 0}
        translated = _snake_case_gemini_thinking_config(config)
        assert translated is not None
        assert translated.get("include_thoughts") is False
        assert translated.get("thinking_budget") == 0

    def test_no_thinking_budget_when_absent(self):
        config = {"includeThoughts": False}
        translated = _snake_case_gemini_thinking_config(config)
        assert translated is not None
        assert translated.get("include_thoughts") is False
        assert "thinking_budget" not in translated


class TestBuildGeminiThinkingConfigEnabled:
    """When reasoning is enabled, thinkingBudget must NOT be set to 0."""

    def test_enabled_does_not_zero_budget(self):
        """When reasoning is enabled, thinkingBudget should not be forced to 0."""
        config = _build_gemini_thinking_config("gemini-2.5-flash", {"enabled": True})
        assert config is not None
        assert config.get("includeThoughts") is True
        assert "thinkingBudget" not in config

    def test_effort_medium_does_not_zero_budget(self):
        config = _build_gemini_thinking_config("gemini-3.6-flash", {"effort": "medium"})
        assert config is not None
        assert config.get("includeThoughts") is True
        assert "thinkingBudget" not in config