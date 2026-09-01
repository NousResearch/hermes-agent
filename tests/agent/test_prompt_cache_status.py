"""Prompt-cache status reporting: a disabled cache must not be silent.

``anthropic_prompt_cache_policy`` can resolve to ``(False, False)`` for a route
that looks perfectly healthy — most commonly a Claude model on an
Anthropic-compatible gateway whose ``model.api_mode`` was never set, which the
URL heuristics cannot detect and which therefore defaults to
``chat_completions``. Nothing then emits a single cache_control breakpoint and
the provider re-bills the entire prompt on every API call.

These tests pin the reporting contract only. They never assert a policy
outcome, so a new branch in the policy cannot break them.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_runtime_helpers import (
    anthropic_prompt_cache_policy,
    prompt_cache_status_summary,
)

GATEWAY = "https://gateway.example.test"


def _agent(**overrides):
    """Minimal destination stub, then resolve the real policy onto it."""
    base = dict(
        provider="custom",
        base_url=GATEWAY,
        api_mode="anthropic_messages",
        model="claude-sonnet-4-5",
        _custom_providers=[],
        _cache_disabled=False,
        verbose_logging=False,
    )
    base.update(overrides)
    agent = SimpleNamespace(**base)
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        anthropic_prompt_cache_policy(agent)
    )
    agent._cache_ttl = None if agent._cache_disabled else "5m"
    return agent


class TestPromptCacheStatusSummary:
    def test_enabled_summary_keeps_the_historical_banner_text(self):
        """The ENABLED wording is unchanged, so operator greps still match."""
        agent = _agent(provider="anthropic", base_url="https://api.anthropic.com")
        assert agent._use_prompt_caching is True
        assert (
            prompt_cache_status_summary(agent)
            == "ENABLED (native Anthropic, 5m TTL)"
        )

    def test_enabled_summary_names_a_third_party_gateway(self):
        agent = _agent()
        assert agent._use_prompt_caching is True
        assert prompt_cache_status_summary(agent) == (
            "ENABLED (Anthropic-compatible endpoint, 5m TTL)"
        )

    def test_unset_api_mode_is_reported_and_names_the_config_key(self):
        """The regression this exists for: silent (False, False) on a Claude route."""
        agent = _agent(api_mode="")
        assert agent._use_prompt_caching is False

        summary = prompt_cache_status_summary(agent)
        assert summary.startswith("DISABLED (")
        assert "api_mode is unset" in summary
        # Actionable: the operator must be told which key to set.
        assert "model.api_mode: anthropic_messages" in summary

    def test_openai_wire_api_mode_is_quoted_back(self):
        agent = _agent(api_mode="chat_completions")
        assert agent._use_prompt_caching is False

        summary = prompt_cache_status_summary(agent)
        assert "api_mode is 'chat_completions'" in summary
        assert "model.api_mode: anthropic_messages" in summary

    def test_non_claude_model_on_the_native_wire_blames_the_model(self):
        """api_mode is already correct here, so the model must be named instead."""
        agent = _agent(model="some-other-model-9")
        assert agent._use_prompt_caching is False

        summary = prompt_cache_status_summary(agent)
        assert "api_mode" not in summary
        assert "'some-other-model-9'" in summary

    def test_operator_disable_is_reported_as_config_not_as_a_route_problem(self):
        agent = _agent(_cache_disabled=True)
        assert agent._use_prompt_caching is False

        summary = prompt_cache_status_summary(agent)
        assert "prompt_caching.cache_ttl" in summary
        # Not a misdiagnosis of the route.
        assert "api_mode" not in summary

    def test_configured_1h_tier_is_reported(self):
        agent = _agent()
        agent._cache_ttl = "1h"
        assert prompt_cache_status_summary(agent).endswith("1h TTL)")

    def test_disabled_reason_is_derived_from_the_policy_not_hardcoded(self):
        """Guard against the explanation drifting away from the real rule.

        The reason is produced by re-running the policy with one input flipped.
        If a future branch makes an unset api_mode cacheable on this route, the
        summary must stop blaming api_mode without anyone editing this helper.
        """
        agent = _agent(api_mode="")
        native_ok, _ = anthropic_prompt_cache_policy(
            agent, api_mode="anthropic_messages"
        )
        blames_api_mode = "api_mode" in prompt_cache_status_summary(agent)
        assert blames_api_mode is native_ok
