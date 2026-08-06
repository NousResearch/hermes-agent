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
