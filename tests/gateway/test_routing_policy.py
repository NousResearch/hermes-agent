"""Tests for thread-local runtime routing policy and manual override commands."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event(text: str = "/route") -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="user-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._session_runtime_overrides = {}
    runner._task_routing_policy = {}
    runner._reasoning_config = {"enabled": True, "effort": "medium"}
    runner._fallback_model = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"}
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key.return_value = "agent:main:discord:group:chan-1"
    return runner


class TestRuntimePolicyResolution:
    def test_resolve_runtime_policy_prefers_thread_model_pin(self):
        runner = _make_runner()
        session_key = runner.session_store._generate_session_key.return_value
        runner._session_runtime_overrides[session_key] = {
            "model": "openai/gpt-5-mini",
            "provider": "openai",
            "reasoning_effort": "low",
        }
        runner._task_routing_policy = {
            "by_task": {
                "code": {"model": "anthropic/claude-sonnet-4.5", "provider": "anthropic"}
            }
        }

        policy = runner._resolve_runtime_policy(session_key=session_key, task_type="code")

        assert policy["model"] == "openai/gpt-5-mini"
        assert policy["provider"] == "openai"
        assert policy["effective_task_type"] == "code"
        assert policy["reasoning_config"] == {"enabled": True, "effort": "low"}

    def test_resolve_runtime_policy_honors_forced_task_type(self):
        runner = _make_runner()
        session_key = runner.session_store._generate_session_key.return_value
        runner._session_runtime_overrides[session_key] = {"forced_task_type": "analysis"}
        runner._task_routing_policy = {
            "by_task": {
                "analysis": {
                    "model": "anthropic/claude-opus-4.6",
                    "provider": "anthropic",
                }
            }
        }

        policy = runner._resolve_runtime_policy(session_key=session_key, task_type="chat")

        assert policy["effective_task_type"] == "analysis"
        assert policy["model"] == "anthropic/claude-opus-4.6"
        assert policy["provider"] == "anthropic"


class TestAskRuntimeOverrides:
    def test_parse_ask_runtime_overrides_with_inline_kv(self):
        overrides, prompt, error = GatewayRunner._parse_ask_runtime_overrides(
            "model=openai/gpt-5-mini reasoning=high Explain the trade-offs"
        )

        assert error is None
        assert prompt == "Explain the trade-offs"
        assert overrides["provider"]
        assert "gpt-5-mini" in overrides["model"]
        assert overrides["reasoning_effort"] == "high"

    def test_format_command_confirmation_includes_emoji_and_details(self):
        msg = GatewayRunner._format_command_confirmation(
            "modelpin",
            "✅ Thread model pinned to `gpt-5.3-codex` (openai).",
        )

        assert "**/modelpin**" in msg
        assert "┊" in msg
        assert "gpt-5.3-codex" in msg

    def test_format_ask_runtime_confirmation_includes_model_and_reasoning(self):
        msg = GatewayRunner._format_ask_runtime_confirmation(
            {
                "effective_task_type": "code",
                "provider": "openai",
                "model": "gpt-5.3-codex",
                "reasoning_config": {"enabled": True, "effort": "high"},
            }
        )

        assert "**/ask**" in msg
        assert "gpt-5.3-codex" in msg
        assert "high" in msg

    def test_parse_ask_runtime_overrides_with_flags(self):
        overrides, prompt, error = GatewayRunner._parse_ask_runtime_overrides(
            "--provider anthropic --model claude-sonnet-4.5 --reasoning low Summarize this"
        )

        assert error is None
        assert prompt == "Summarize this"
        assert overrides == {
            "provider": "anthropic",
            "model": "claude-sonnet-4.5",
            "reasoning_effort": "low",
        }

    def test_parse_ask_runtime_overrides_rejects_invalid_reasoning(self):
        overrides, prompt, error = GatewayRunner._parse_ask_runtime_overrides(
            "reasoning=turbo Explain this"
        )

        assert overrides == {}
        assert prompt == ""
        assert "invalid reasoning" in error.lower()


class TestManualOverrideCommands:
    @pytest.mark.asyncio
    async def test_modelpin_sets_and_clears_override(self):
        runner = _make_runner()
        session_key = runner.session_store._generate_session_key.return_value

        set_msg = await runner._handle_modelpin_command(_make_event("/modelpin gpt-5-mini"))
        assert "pinned" in set_msg.lower()
        assert runner._session_runtime_overrides[session_key]["provider"]
        assert runner._session_runtime_overrides[session_key]["model"] == "gpt-5-mini"

        clear_msg = await runner._handle_modelpin_command(_make_event("/modelpin clear"))
        assert "cleared" in clear_msg.lower()
        assert "provider" not in runner._session_runtime_overrides[session_key]
        assert "model" not in runner._session_runtime_overrides[session_key]

    @pytest.mark.asyncio
    async def test_reasoning_command_sets_and_clears_override(self):
        runner = _make_runner()
        session_key = runner.session_store._generate_session_key.return_value

        msg = await runner._handle_reasoning_command(_make_event("/reasoning high"))
        assert "high" in msg
        assert runner._session_runtime_overrides[session_key]["reasoning_effort"] == "high"

        clear_msg = await runner._handle_reasoning_command(_make_event("/reasoning default"))
        assert "cleared" in clear_msg.lower()
        assert "reasoning_effort" not in runner._session_runtime_overrides[session_key]

    @pytest.mark.asyncio
    async def test_route_command_sets_and_rejects_invalid_task_types(self):
        runner = _make_runner()
        session_key = runner.session_store._generate_session_key.return_value

        ok_msg = await runner._handle_route_command(_make_event("/route code"))
        assert "code" in ok_msg
        assert runner._session_runtime_overrides[session_key]["forced_task_type"] == "code"

        bad_msg = await runner._handle_route_command(_make_event("/route nonsense"))
        assert "invalid" in bad_msg.lower()

    @pytest.mark.asyncio
    async def test_runtime_command_reports_overrides_and_resolved_policy(self):
        runner = _make_runner()
        session_key = runner.session_store._generate_session_key.return_value
        runner._session_runtime_overrides[session_key] = {
            "model": "gpt-5-mini",
            "provider": "openai",
            "reasoning_effort": "low",
            "forced_task_type": "analysis",
        }

        runtime_msg = await runner._handle_runtime_command(_make_event("/runtime"))

        assert "thread runtime policy" in runtime_msg.lower()
        assert "gpt-5-mini" in runtime_msg
        assert "analysis" in runtime_msg
        assert "reasoning" in runtime_msg.lower()
