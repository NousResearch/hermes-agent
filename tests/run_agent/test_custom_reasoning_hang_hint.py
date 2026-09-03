"""Tests for the custom-provider reasoning_effort stale-hang hint (#100841).

A `provider: custom` endpoint (vLLM / llama.cpp / SGLang) that honours a
forwarded `reasoning_effort` as extended thinking can stall for minutes with
no error, surfacing only as a generic "Non-streaming API call timed out".
The hint in the timeout text must name reasoning_effort as the likely cause
so operators don't have to rediscover it through timing experiments.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="qwen3.8-27b",
        provider="custom",
        api_key="sk-dummy",
        base_url="http://127.0.0.1:8000/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


@pytest.fixture(autouse=True)
def _isolate_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")


def test_hint_fires_for_custom_provider_with_reasoning_effort(tmp_path):
    agent = _make_agent(tmp_path)
    hint = agent._custom_reasoning_hang_hint(
        {"model": "qwen3.8-27b", "reasoning_effort": "medium"}
    )
    assert hint is not None
    assert "reasoning_effort" in hint
    assert "'medium'" in hint
    assert "agent.reasoning_effort" in hint
    assert "100841" in hint


def test_hint_silent_when_no_effort_sent(tmp_path):
    agent = _make_agent(tmp_path)
    assert agent._custom_reasoning_hang_hint({"model": "qwen3.8-27b"}) is None
    assert agent._custom_reasoning_hang_hint(
        {"model": "qwen3.8-27b", "reasoning_effort": "none"}
    ) is None
    assert agent._custom_reasoning_hang_hint(
        {"model": "qwen3.8-27b", "reasoning_effort": ""}
    ) is None
    assert agent._custom_reasoning_hang_hint(None) is None
    assert agent._custom_reasoning_hang_hint("not-a-dict") is None


def test_hint_silent_for_non_custom_providers(tmp_path):
    agent = _make_agent(tmp_path, provider="openrouter",
                        base_url="https://openrouter.ai/api/v1")
    assert agent._custom_reasoning_hang_hint(
        {"model": "qwen3.8-27b", "reasoning_effort": "medium"}
    ) is None


def test_hint_reads_extra_body_effort(tmp_path):
    """Some transports nest reasoning_effort under extra_body."""
    agent = _make_agent(tmp_path)
    hint = agent._custom_reasoning_hang_hint(
        {"model": "qwen3.8-27b", "extra_body": {"reasoning_effort": "high"}}
    )
    assert hint is not None and "'high'" in hint


def test_stale_resolver_prefers_codex_hint_when_both_apply(tmp_path):
    """Codex silent-reject is the sharper diagnosis; it must win the slot."""
    from agent.chat_completion_helpers import _stale_hang_hint
    agent = _make_agent(
        tmp_path, provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", model="gpt-5.5",
    )
    agent.api_mode = "codex_responses"
    hint = _stale_hang_hint(
        agent, {"model": "gpt-5.5", "reasoning_effort": "medium"}
    )
    assert hint is not None
    assert "backend-api/codex" in hint


def test_stale_resolver_falls_through_to_custom_reasoning(tmp_path):
    from agent.chat_completion_helpers import _stale_hang_hint
    agent = _make_agent(tmp_path)
    hint = _stale_hang_hint(
        agent, {"model": "qwen3.8-27b", "reasoning_effort": "medium"}
    )
    assert hint is not None and "reasoning_effort" in hint


def test_stale_resolver_returns_none_without_matching_pattern(tmp_path):
    from agent.chat_completion_helpers import _stale_hang_hint
    agent = _make_agent(tmp_path)
    assert _stale_hang_hint(agent, {"model": "qwen3.8-27b"}) is None
