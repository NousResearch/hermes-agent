"""Tests for gateway /context command composition reporting."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


SK = "agent:main:telegram:private:12345"


def _make_runner(session_key, agent=None, cached_agent=None, history=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(session_id="sid")
    runner.session_store.load_transcript.return_value = history or []
    runner._session_key_for_source = MagicMock(return_value=session_key)

    if agent is not None:
        runner._running_agents[session_key] = agent
    if cached_agent is not None:
        runner._agent_cache[session_key] = (cached_agent, "sig")

    return runner


def _make_agent():
    system_parts = {
        "stable": "base guidance",
        "context": "AGENTS.md instructions",
        "volatile": "Conversation started: Tuesday, May 26, 2026",
    }
    messages = [
        {"role": "user", "content": "inspect context"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_terminal",
                    "function": {"name": "terminal", "arguments": '{"command": "pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_terminal", "content": "/tmp/project\n"},
    ]
    return SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        tools=[{"type": "function", "function": {"name": "terminal", "description": "run commands"}}],
        context_compressor=SimpleNamespace(
            context_length=272_000,
            threshold_tokens=217_600,
            compression_count=0,
            last_prompt_tokens=0,
        ),
        _cached_system_prompt="\n\n".join(system_parts.values()),
        _session_messages=messages,
        _build_system_prompt_parts=lambda _system_message=None: system_parts,
    )


class TestContextCommand:
    @pytest.mark.asyncio
    async def test_cached_agent_shows_context_report(self):
        runner = _make_runner(SK, cached_agent=_make_agent())
        result = await runner._handle_context_command(SimpleNamespace(source="src"))

        assert "🧠 Context Window" in result
        assert "Model: openai-codex:gpt-5.5" in result
        assert "📦 Buckets" in result
        assert "System prompt" in result
        assert "Tools schema" in result
        assert "📎 Heaviest tool results" in result
        assert "tok (" in result
        assert "[#" not in result
        assert "terminal" in result

    @pytest.mark.asyncio
    async def test_history_without_agent_returns_basic_estimate(self):
        runner = _make_runner(SK, history=[{"role": "user", "content": "hello"}])
        result = await runner._handle_context_command(SimpleNamespace(source="src"))

        assert "🧠 Context Window" in result
        assert "Estimated messages:" in result
        assert "Detailed system/tool breakdown is available after the first agent turn." in result
