import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/goal", platform=Platform.SLACK, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._cleanup_agent_resources = MagicMock()
    runner._resolve_session_agent_runtime = MagicMock(return_value=("test-model", {"api_key": "test-key", "provider": "test"}))
    runner._resolve_session_reasoning_config = MagicMock(return_value=None)
    runner._load_service_tier = MagicMock(return_value=None)
    runner._resolve_turn_agent_config = MagicMock(return_value={"model": "test-model", "runtime": {"api_key": "test-key", "provider": "test"}})

    async def run_immediately(func):
        return func()

    runner._run_in_executor_with_context = AsyncMock(side_effect=run_immediately)
    return runner


class TestHandleGoalCommand:
    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        runner = _make_runner()
        result = await runner._handle_goal_command(_make_event("/goal"))
        assert "Usage:" in result
        assert "/goal" in result

    @pytest.mark.asyncio
    async def test_valid_prompt_starts_goal_task(self, monkeypatch):
        runner = _make_runner()
        monkeypatch.setenv("HERMES_GOAL_MAX_LOOPS", "2")
        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_goal_command(_make_event("/goal finish AutomateSTIG batch"))

        assert "🎯" in result
        assert "Goal loop started" in result
        assert "goal_" in result
        assert "Max supervisor loops: 2" in result
        assert "finish AutomateSTIG batch" in result
        assert len(created_tasks) == 1

    @pytest.mark.asyncio
    async def test_long_prompt_is_truncated(self):
        runner = _make_runner()
        long_prompt = "A" * 100
        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            result = await runner._handle_goal_command(_make_event(f"/goal {long_prompt}"))
        assert "..." in result
        assert long_prompt not in result

    @pytest.mark.asyncio
    async def test_goal_task_expands_nested_skill_command_before_worker_runs(self, monkeypatch):
        runner = _make_runner()
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.extract_media = MagicMock(side_effect=lambda text: ([], text))
        adapter.extract_images = MagicMock(side_effect=lambda text: ([], text))
        runner.adapters[Platform.SLACK] = adapter
        sent_worker_prompts = []

        def fake_resolve(command):
            assert command == "automatestig-disa-coverage-batch"
            return "/automatestig-disa-coverage-batch"

        def fake_build(cmd_key, user_instruction="", task_id=None, runtime_note=""):
            assert cmd_key == "/automatestig-disa-coverage-batch"
            assert user_instruction == "keep batching"
            assert task_id == "goal_test"
            assert "nested skill" in runtime_note
            return "[IMPORTANT: AutomateSTIG skill loaded]\n" + user_instruction

        monkeypatch.setattr("agent.skill_commands.resolve_skill_command_key", fake_resolve)
        monkeypatch.setattr("agent.skill_commands.build_skill_invocation_message", fake_build)
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_args, **_kwargs: {"terminal", "file"})

        class FakeAgent:
            def __init__(self, *args, enabled_toolsets=None, **kwargs):
                self.enabled_toolsets = enabled_toolsets

            def run_conversation(self, user_message, task_id=None):
                if self.enabled_toolsets:
                    sent_worker_prompts.append(user_message)
                    return {"final_response": "worker done"}
                return {"final_response": "COMPLETE: worker proved the AutomateSTIG batch is done"}

        with patch("run_agent.AIAgent", FakeAgent):
            await runner._run_goal_task(
                "/automatestig-disa-coverage-batch keep batching",
                _make_event().source,
                "goal_test",
            )

        assert sent_worker_prompts
        assert "AutomateSTIG skill loaded" in sent_worker_prompts[0]
        assert "keep batching" in sent_worker_prompts[0]
        assert "/automatestig-disa-coverage-batch keep batching" not in sent_worker_prompts[0]
        adapter.send.assert_awaited_once()
        sent_text = adapter.send.await_args.args[1]
        assert "Loaded skill: automatestig-disa-coverage-batch" in sent_text
