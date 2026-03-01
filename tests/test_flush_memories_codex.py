"""Tests for flush_memories() working correctly across all provider modes.

Catches the bug where Codex mode called chat.completions.create on a
Responses-only client, which would fail silently or with a 404.
"""

import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, call

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key", "test")
        self.base_url = kwargs.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter"):
    """Build an AIAgent with mocked internals, ready for flush_memories testing."""
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [
        {
            "type": "function",
            "function": {
                "name": "memory",
                "description": "Manage memories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    ])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)

    agent = run_agent.AIAgent(
        api_key="test-key",
        base_url="https://test.example.com/v1",
        provider=provider,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    # Give it a valid memory store
    agent._memory_store = MagicMock()
    agent._memory_flush_min_turns = 1
    agent._user_turn_count = 5
    return agent


def _chat_response_with_memory_call():
    """Simulated chat completions response with a memory tool call."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    function=SimpleNamespace(
                        name="memory",
                        arguments=json.dumps({
                            "action": "add",
                            "target": "notes",
                            "content": "User prefers dark mode.",
                        }),
                    ),
                )],
            ),
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


class TestFlushMemoriesUsesAuxiliaryClient:
    """When an auxiliary client is available, flush_memories should use it
    instead of self.client -- especially critical in Codex mode."""

    def test_flush_uses_auxiliary_when_available(self, monkeypatch):
        agent = _make_agent(monkeypatch, api_mode="codex_responses", provider="openai-codex")

        mock_aux_client = MagicMock()
        mock_aux_client.chat.completions.create.return_value = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_aux_client, "gpt-4o-mini")):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Remember this"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value="Saved.") as mock_memory:
                agent.flush_memories(messages)

        mock_aux_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_aux_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4o-mini" or call_kwargs[1].get("model") == "gpt-4o-mini"

    def test_flush_uses_main_client_when_no_auxiliary(self, monkeypatch):
        """Non-Codex mode with no auxiliary falls back to self.client."""
        agent = _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter")
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(None, None)):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "Save this"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value="Saved."):
                agent.flush_memories(messages)

        agent.client.chat.completions.create.assert_called_once()

    def test_flush_executes_memory_tool_calls(self, monkeypatch):
        """Verify that memory tool calls from the flush response actually get executed."""
        agent = _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter")

        mock_aux_client = MagicMock()
        mock_aux_client.chat.completions.create.return_value = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_aux_client, "gpt-4o-mini")):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Note this"},
            ]
            with patch("tools.memory_tool.memory_tool", return_value="Saved.") as mock_memory:
                agent.flush_memories(messages)

        mock_memory.assert_called_once()
        call_kwargs = mock_memory.call_args
        assert call_kwargs.kwargs["action"] == "add"
        assert call_kwargs.kwargs["target"] == "notes"
        assert "dark mode" in call_kwargs.kwargs["content"]

    def test_flush_strips_artifacts_from_messages(self, monkeypatch):
        """After flush, the flush prompt and any response should be removed from messages."""
        agent = _make_agent(monkeypatch, api_mode="chat_completions", provider="openrouter")

        mock_aux_client = MagicMock()
        mock_aux_client.chat.completions.create.return_value = _chat_response_with_memory_call()

        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_aux_client, "gpt-4o-mini")):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Remember X"},
            ]
            original_len = len(messages)
            with patch("tools.memory_tool.memory_tool", return_value="Saved."):
                agent.flush_memories(messages)

        # Messages should not grow from the flush
        assert len(messages) <= original_len
        # No flush sentinel should remain
        for msg in messages:
            assert "_flush_sentinel" not in msg


class TestFlushMemoriesCodexFallback:
    """When no auxiliary client exists and we're in Codex mode, flush should
    use the Codex Responses API path instead of chat.completions."""

    def test_codex_mode_no_aux_uses_responses_api(self, monkeypatch):
        agent = _make_agent(monkeypatch, api_mode="codex_responses", provider="openai-codex")

        codex_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="memory",
                    arguments=json.dumps({
                        "action": "add",
                        "target": "notes",
                        "content": "Codex flush test",
                    }),
                ),
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=10, total_tokens=60),
            status="completed",
            model="gpt-5-codex",
        )

        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(None, None)), \
             patch.object(agent, "_run_codex_stream", return_value=codex_response) as mock_stream, \
             patch.object(agent, "_build_api_kwargs") as mock_build, \
             patch("tools.memory_tool.memory_tool", return_value="Saved.") as mock_memory:
            mock_build.return_value = {
                "model": "gpt-5-codex",
                "instructions": "test",
                "input": [],
                "tools": [],
                "max_output_tokens": 4096,
            }
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Save this"},
            ]
            agent.flush_memories(messages)

        mock_stream.assert_called_once()
        mock_memory.assert_called_once()
        assert mock_memory.call_args.kwargs["content"] == "Codex flush test"
