"""Tests for Codex economy mode — prompt_builder constant + builder function,
system_prompt integration via build_system_prompt_parts."""

import types
from unittest.mock import patch

import pytest

from agent.prompt_builder import CODEX_ECONOMY_GUIDANCE, build_codex_economy_prompt, _format_economy_guidance


# =========================================================================
# build_codex_economy_prompt unit tests
# =========================================================================


class TestBuildCodexEconomyPrompt:
    def test_disabled_returns_empty(self):
        assert build_codex_economy_prompt(enabled=False) == ""

    def test_all_defaults_return_empty(self):
        assert build_codex_economy_prompt() == ""

    def test_enabled_returns_guidance(self):
        result = build_codex_economy_prompt(enabled=True)
        assert result == CODEX_ECONOMY_GUIDANCE
        assert "context_efficiency" in result
        assert "search_files" in result

    def test_enabled_overrides_no_provider(self):
        # enabled=True should return guidance even without provider/model.
        result = build_codex_economy_prompt(enabled=True, provider="", model="")
        assert result == CODEX_ECONOMY_GUIDANCE

    def test_auto_for_openai_codex_provider(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=True,
            provider="openai-codex",
        )
        assert result == CODEX_ECONOMY_GUIDANCE

    def test_auto_for_xai_oauth_provider(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=True,
            provider="xai-oauth",
        )
        assert result == CODEX_ECONOMY_GUIDANCE

    def test_auto_for_codex_responses_api_mode(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=True,
            api_mode="codex_responses",
        )
        assert result == CODEX_ECONOMY_GUIDANCE

    def test_auto_for_codex_in_model_name(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=True,
            model="openai/codex-mini-latest",
        )
        assert result == CODEX_ECONOMY_GUIDANCE

    def test_auto_off_codex_provider_no_match(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=False,
            provider="openai-codex",
        )
        assert result == ""

    def test_auto_on_non_codex_provider_no_match(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=True,
            provider="anthropic",
            api_mode="anthropic_messages",
            model="claude-opus-4-7",
        )
        assert result == ""

    def test_guidance_preserves_quality_language(self):
        assert "Do NOT skip verification" in CODEX_ECONOMY_GUIDANCE
        assert "final tests" in CODEX_ECONOMY_GUIDANCE
        assert "non-negotiable" in CODEX_ECONOMY_GUIDANCE

    def test_guidance_preserves_parallel_language(self):
        assert "parallel execution" in CODEX_ECONOMY_GUIDANCE
        assert "NOT serialize" in CODEX_ECONOMY_GUIDANCE

    def test_guidance_mentions_codex_final(self):
        assert "Codex" in CODEX_ECONOMY_GUIDANCE

    def test_guidance_mentions_search_files(self):
        assert "search_files" in CODEX_ECONOMY_GUIDANCE

    def test_default_budgets_in_guidance(self):
        assert "8" in CODEX_ECONOMY_GUIDANCE
        assert "400" in CODEX_ECONOMY_GUIDANCE

    def test_custom_budgets_appear_in_prompt(self):
        result = build_codex_economy_prompt(enabled=True, max_changed_files=5, max_diff_lines=200)
        assert "5" in result
        assert "200" in result
        assert "8" not in result or "400" not in result  # custom values replace defaults

    def test_custom_budgets_auto_mode(self):
        result = build_codex_economy_prompt(
            auto_for_openai_codex=True,
            provider="openai-codex",
            max_changed_files=12,
            max_diff_lines=600,
        )
        assert "12" in result
        assert "600" in result

    def test_format_economy_guidance_injects_values(self):
        result = _format_economy_guidance(max_changed_files=3, max_diff_lines=150)
        assert "3" in result
        assert "150" in result
        assert "context_efficiency" in result
        assert "search_files" in result

    def test_format_economy_guidance_defaults_match_constant(self):
        assert _format_economy_guidance() == CODEX_ECONOMY_GUIDANCE


# =========================================================================
# system_prompt integration tests
# =========================================================================


def _make_agent(**kwargs):
    """Build a minimal agent namespace for build_system_prompt_parts."""
    agent = types.SimpleNamespace(
        model="claude-opus-4-7",
        provider="anthropic",
        api_mode="anthropic_messages",
        platform="",
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=set(),
        _tool_use_enforcement=False,
        _kanban_worker_guidance=None,
        _memory_store=None,
        _memory_manager=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        pass_session_id=False,
        session_id=None,
        _codex_economy_enabled=False,
        _codex_economy_auto_openai_codex=False,
        _codex_economy_max_changed_files=8,
        _codex_economy_max_diff_lines=400,
    )
    for k, v in kwargs.items():
        setattr(agent, k, v)
    return agent


class TestSystemPromptIntegration:
    def test_economy_absent_when_disabled(self):
        from agent.system_prompt import build_system_prompt_parts

        agent = _make_agent(
            _codex_economy_enabled=False,
            _codex_economy_auto_openai_codex=False,
        )
        parts = build_system_prompt_parts(agent)
        assert "context_efficiency" not in parts["stable"]

    def test_economy_present_when_enabled(self):
        from agent.system_prompt import build_system_prompt_parts

        agent = _make_agent(_codex_economy_enabled=True)
        parts = build_system_prompt_parts(agent)
        assert "context_efficiency" in parts["stable"]

    def test_economy_present_when_auto_and_codex_provider(self):
        from agent.system_prompt import build_system_prompt_parts

        agent = _make_agent(
            _codex_economy_enabled=False,
            _codex_economy_auto_openai_codex=True,
            provider="openai-codex",
            api_mode="codex_responses",
            model="codex-mini-latest",
        )
        parts = build_system_prompt_parts(agent)
        assert "context_efficiency" in parts["stable"]

    def test_economy_absent_for_anthropic_auto(self):
        from agent.system_prompt import build_system_prompt_parts

        agent = _make_agent(
            _codex_economy_enabled=False,
            _codex_economy_auto_openai_codex=True,
            provider="anthropic",
            api_mode="anthropic_messages",
            model="claude-opus-4-7",
        )
        parts = build_system_prompt_parts(agent)
        assert "context_efficiency" not in parts["stable"]

    def test_economy_in_stable_not_volatile(self):
        from agent.system_prompt import build_system_prompt_parts

        agent = _make_agent(_codex_economy_enabled=True)
        parts = build_system_prompt_parts(agent)
        assert "context_efficiency" in parts["stable"]
        assert "context_efficiency" not in parts["volatile"]
