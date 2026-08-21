"""Tests for AIAgent provider=meta host-mandated api_mode."""

import pytest


def test_agent_init_meta_base_url_implies_codex_responses():
    from run_agent import AIAgent

    agent = AIAgent(
        provider="meta",
        base_url="https://api.meta.ai/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        api_mode=None,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"
    assert agent.provider == "meta"


def test_agent_init_explicit_chat_wins_over_mandate():
    from run_agent import AIAgent

    agent = AIAgent(
        provider="meta",
        base_url="https://api.meta.ai/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "chat_completions"
    assert agent.provider == "meta"


def test_agent_init_meta_provider_stays_meta():
    from run_agent import AIAgent

    agent = AIAgent(
        provider="meta",
        base_url="https://api.meta.ai/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.provider == "meta"
    assert agent.provider != "custom"


def test_agent_init_custom_with_meta_url_also_mandates():
    from run_agent import AIAgent

    agent = AIAgent(
        provider="custom",
        base_url="https://api.meta.ai/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"


def test_agent_init_meta_url_case_insensitive():
    from run_agent import AIAgent

    agent = AIAgent(
        provider="meta",
        base_url="https://API.META.AI/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"


def test_agent_init_anthropic_url_implies_provider_and_api_mode():
    """Regression for host-mandated api_mode shadowing provider-slug rewrite (#63425).

    When provider=None and base_url is api.anthropic.com, the elif chain must
    still rewrite provider to "anthropic" before the credential-pool check.
    The host-mandated fallback moved to the bottom of the cascade (fix/meta-prompt-caching)
    ensures this; the old top-position mandated branch set api_mode without the slug rewrite
    and caused credential-pool validation to discard anthropic-scoped pools.
    """
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent

    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()), patch(
        "agent.anthropic_adapter._is_oauth_token", return_value=False
    ):
        agent = AIAgent(
            provider=None,
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test-anthropic",
            model="claude-sonnet-4-5",
            api_mode=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.provider == "anthropic"
    assert agent.api_mode == "anthropic_messages"


def test_agent_init_anthropic_url_preserves_credential_pool():
    """Anthropic-scoped credential pool must survive provider auto-detection.

    Mirrors tests/run_agent/test_63425_credential_pool_auto_detect.py at the
    public AIAgent surface (provider=None URL detection).
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from run_agent import AIAgent

    pool = SimpleNamespace(provider="anthropic")

    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()), patch(
        "agent.anthropic_adapter._is_oauth_token", return_value=False
    ):
        agent = AIAgent(
            provider=None,
            base_url="https://api.anthropic.com/v1",
            api_key="sk-test-anthropic",
            model="claude-sonnet-4-5",
            api_mode=None,
            credential_pool=pool,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.provider == "anthropic"
    assert agent.api_mode == "anthropic_messages"
    assert agent._credential_pool is pool


def test_agent_init_meta_url_without_provider_implies_codex_responses():
    from run_agent import AIAgent

    agent = AIAgent(
        provider=None,
        base_url="https://api.meta.ai/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        api_mode=None,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.api_mode == "codex_responses"


def test_agent_init_meta_provider_without_meta_url_falls_through_to_chat_completions(monkeypatch):
    """`provider="meta"` without an api.meta.ai base_url falls through to chat_completions.

    The Meta wire protocol is URL-driven by design so custom OpenAI-compatible
    endpoints configured under `providers.meta` are not broken by forcing Responses API.
    """
    from run_agent import AIAgent

    monkeypatch.setenv("META_API_KEY", "sk-test-meta")

    agent_empty = AIAgent(
        provider="meta",
        base_url="",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        api_mode=None,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent_empty.api_mode == "chat_completions"
    assert agent_empty.provider == "meta"

    agent_custom = AIAgent(
        provider="meta",
        base_url="https://custom.meta.endpoint.internal/v1",
        api_key="sk-test-meta",
        model="muse-spark-1.2",
        api_mode=None,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent_custom.api_mode == "chat_completions"
    assert agent_custom.provider == "meta"


