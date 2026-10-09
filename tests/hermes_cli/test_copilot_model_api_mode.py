"""Tests for Copilot model-dependent canonical route policy."""

from __future__ import annotations

from providers.routing import InvocationRequest, resolve_invocation_route


def _mode(model: str) -> str:
    return resolve_invocation_route(
        InvocationRequest(provider="copilot", model=model)
    ).api_mode


def test_copilot_claude_stays_on_chat_completions():
    assert _mode("claude-opus-4.8") == "chat_completions"


def test_copilot_gpt5_uses_responses_except_mini():
    assert _mode("gpt-5.5") == "codex_responses"
    assert _mode("gpt-5-mini") == "chat_completions"
