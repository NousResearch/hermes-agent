"""Test validation error prevention for strict APIs (Fireworks, etc.)"""

import sys
import types
from unittest.mock import patch, MagicMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, provider, api_mode="chat_completions", base_url="https://openrouter.ai/api/v1"):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search", "terminal"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test",
        base_url=base_url,
        provider=provider,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


class TestStrictApiValidation:
    """Verify tool_call field sanitization prevents 400 errors on strict APIs."""

    def test_fireworks_compatible_messages_after_sanitization(self, monkeypatch):
        """Messages should be Fireworks-compatible after sanitization."""
        agent = _make_agent(monkeypatch, "openrouter")
        agent.api_mode = "chat_completions"  # Fireworks uses chat completions

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Checking now.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "call_id": "call_123",  # Codex-only field
                        "response_item_id": "fc_123",  # Codex-only field
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "/tmp"},
        ]

        # After _build_api_kwargs, Codex fields should be stripped
        kwargs = agent._build_api_kwargs(messages)

        assistant_msg = kwargs["messages"][1]
        tool_call = assistant_msg["tool_calls"][0]

        # Fireworks rejects these fields
        assert "call_id" not in tool_call
        assert "response_item_id" not in tool_call
        # Standard fields should remain
        assert tool_call["id"] == "call_123"
        assert tool_call["function"]["name"] == "terminal"

    def test_codex_preserves_fields_for_replay(self, monkeypatch):
        """Codex mode should preserve fields for Responses API replay."""
        agent = _make_agent(monkeypatch, "openrouter")
        agent.api_mode = "codex_responses"

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Checking now.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "call_id": "call_123",
                        "response_item_id": "fc_123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
        ]

        # In Codex mode, original messages should NOT be mutated
        assert messages[1]["tool_calls"][0]["call_id"] == "call_123"
        assert messages[1]["tool_calls"][0]["response_item_id"] == "fc_123"

    def test_sanitize_method_with_fireworks_provider(self, monkeypatch):
        """Simulating Fireworks provider should trigger sanitization."""
        agent = _make_agent(
            monkeypatch,
            "fireworks",
            api_mode="chat_completions",
            base_url="https://api.fireworks.ai/inference/v1"
        )

        # Should sanitize for Fireworks (chat_completions mode)
        assert agent._should_sanitize_tool_calls() is True

    def test_no_sanitize_for_codex_responses(self, monkeypatch):
        """Codex responses mode should NOT sanitize."""
        agent = _make_agent(
            monkeypatch,
            "openai",
            api_mode="codex_responses",
            base_url="https://api.openai.com/v1"
        )

        # Should NOT sanitize for Codex
        assert agent._should_sanitize_tool_calls() is False


class TestGroqReasoningContentStrip:
    """Regression for #11089.

    Groq's OpenAI-compatible endpoint rejects ``reasoning_content`` on
    role:assistant with HTTP 400 ``property 'reasoning_content' is
    unsupported``. Hermes unconditionally copies a persisted ``reasoning``
    field onto outbound assistant messages as ``reasoning_content`` (needed
    for Moonshot/Novita/OpenRouter multi-turn reasoning), which breaks users
    who point ``provider: custom`` at Groq after any thinking-enabled turn.

    ``_build_api_kwargs`` must strip ``reasoning_content`` from outbound
    assistant messages when the base URL targets Groq, while leaving the
    internal message history untouched so the reasoning context survives
    a later switch back to a reasoning-aware provider.
    """

    def test_groq_rejects_reasoning_content_flag(self, monkeypatch):
        groq_agent = _make_agent(
            monkeypatch,
            "custom",
            api_mode="chat_completions",
            base_url="https://api.groq.com/openai/v1",
        )
        assert groq_agent._provider_rejects_reasoning_content() is True

        or_agent = _make_agent(
            monkeypatch,
            "openrouter",
            api_mode="chat_completions",
            base_url="https://openrouter.ai/api/v1",
        )
        assert or_agent._provider_rejects_reasoning_content() is False

    def test_groq_strips_reasoning_content_from_outgoing_messages(self, monkeypatch):
        """Outgoing Groq payload must not carry ``reasoning_content``."""
        agent = _make_agent(
            monkeypatch,
            "custom",
            api_mode="chat_completions",
            base_url="https://api.groq.com/openai/v1",
        )

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "User greeted me; respond warmly.",
            },
            {"role": "user", "content": "what's 2+2?"},
        ]

        kwargs = agent._build_api_kwargs(messages)

        assistant_msg = kwargs["messages"][1]
        # Groq would 400 on this — must be stripped.
        assert "reasoning_content" not in assistant_msg
        # Regular content preserved.
        assert assistant_msg["content"] == "Hello!"
        assert assistant_msg["role"] == "assistant"

    def test_groq_preserves_internal_history(self, monkeypatch):
        """Stripping the outgoing copy must not mutate the caller's list."""
        agent = _make_agent(
            monkeypatch,
            "custom",
            api_mode="chat_completions",
            base_url="https://api.groq.com/openai/v1",
        )

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "internal thought",
            },
        ]

        agent._build_api_kwargs(messages)

        # Internal history untouched — a later switch back to a reasoning-
        # aware provider still has the context.
        assert messages[1]["reasoning_content"] == "internal thought"

    def test_non_groq_preserves_reasoning_content(self, monkeypatch):
        """OpenRouter (and other reasoning-aware providers) still receive
        ``reasoning_content`` so multi-turn reasoning context is preserved."""
        agent = _make_agent(
            monkeypatch,
            "openrouter",
            api_mode="chat_completions",
            base_url="https://openrouter.ai/api/v1",
        )

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "User greeted me.",
            },
        ]

        kwargs = agent._build_api_kwargs(messages)
        assistant_msg = kwargs["messages"][1]

        assert assistant_msg["reasoning_content"] == "User greeted me."

    def test_groq_strips_alongside_codex_fields(self, monkeypatch):
        """When both Codex fields and reasoning_content are present, both
        are stripped in a single deepcopy pass."""
        agent = _make_agent(
            monkeypatch,
            "custom",
            api_mode="chat_completions",
            base_url="https://api.groq.com/openai/v1",
        )

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Checking.",
                "reasoning_content": "think",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "call_id": "call_1",
                        "response_item_id": "fc_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
        ]

        kwargs = agent._build_api_kwargs(messages)
        asst = kwargs["messages"][1]

        assert "reasoning_content" not in asst
        tc = asst["tool_calls"][0]
        assert "call_id" not in tc
        assert "response_item_id" not in tc
        assert tc["id"] == "call_1"

    def test_groq_no_reasoning_content_is_a_noop(self, monkeypatch):
        """When there is no ``reasoning_content`` to strip, the outgoing
        messages should be the exact same list (no unnecessary deepcopy)."""
        agent = _make_agent(
            monkeypatch,
            "custom",
            api_mode="chat_completions",
            base_url="https://api.groq.com/openai/v1",
        )

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        kwargs = agent._build_api_kwargs(messages)

        # No strip needed → outgoing list is the original reference.
        assert kwargs["messages"] is messages
