"""Tests for /swarm gateway slash command.

Covers detached foreman launch/control behavior while the parent chat remains
free to continue.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/swarm", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890", thread_id=None):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        thread_id=thread_id,
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="***",
                home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="67890", name="Home"),
            )
        }
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()

    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = MagicMock(session_id="parent-session")
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner


class TestHandleSwarmCommand:
    @pytest.mark.asyncio
    async def test_start_without_prompt_shows_usage(self):
        runner = _make_runner()
        event = _make_event(text="/swarm start")
        result = await runner._handle_swarm_command(event)
        assert "Usage:" in result
        assert "start <prompt>" in result

    @pytest.mark.asyncio
    async def test_launch_starts_detached_task(self):
        runner = _make_runner()
        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 7}}), \
             patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/swarm Investigate auth regressions")
            result = await runner._handle_swarm_command(event)

        assert "Detached swarm started" in result
        assert "Main Sub is running in the background" in result
        assert "Worker budget: up to 7" in result
        assert "swarm_" in result
        assert len(created_tasks) == 1

    @pytest.mark.asyncio
    async def test_stop_without_matching_foreman_reports_none(self):
        runner = _make_runner()
        with patch("tools.delegate_tool.list_active_subagents", return_value=[]):
            result = await runner._handle_swarm_command(_make_event(text="/swarm stop"))
        assert "No detached swarm foreman is active for this chat" in result

    @pytest.mark.asyncio
    async def test_stop_interrupts_single_matching_foreman(self):
        runner = _make_runner()
        rows = [{
            "subagent_id": "swarm_123",
            "kind": "detached_foreman",
            "source_platform": "telegram",
            "source_chat_id": "67890",
            "source_thread_id": "",
        }]
        with patch("tools.delegate_tool.list_active_subagents", return_value=rows), \
             patch("tools.delegate_tool.interrupt_subagent", return_value=True) as interrupt:
            result = await runner._handle_swarm_command(_make_event(text="/swarm stop"))
        interrupt.assert_called_once_with("swarm_123")
        assert "Swarm stop requested" in result

    @pytest.mark.asyncio
    async def test_explicit_stop_target_is_used(self):
        runner = _make_runner()
        rows = [{
            "subagent_id": "swarm_abc",
            "kind": "detached_foreman",
            "source_platform": "telegram",
            "source_chat_id": "67890",
            "source_thread_id": "",
        }]
        with patch("tools.delegate_tool.list_active_subagents", return_value=rows), \
             patch("tools.delegate_tool.interrupt_subagent", return_value=True) as interrupt:
            result = await runner._handle_swarm_command(_make_event(text="/swarm stop swarm_abc"))
        interrupt.assert_called_once_with("swarm_abc")
        assert "swarm_abc" in result

    @pytest.mark.asyncio
    async def test_explicit_stop_rejects_other_chat_foreman(self):
        runner = _make_runner()
        rows = [{
            "subagent_id": "swarm_other",
            "kind": "detached_foreman",
            "source_platform": "telegram",
            "source_chat_id": "different-chat",
            "source_thread_id": "",
        }]
        with patch("tools.delegate_tool.list_active_subagents", return_value=rows):
            result = await runner._handle_swarm_command(_make_event(text="/swarm stop swarm_other"))
        assert "belongs to a different chat" in result

    @pytest.mark.asyncio
    async def test_stop_with_multiword_arg_shows_usage_and_does_not_launch(self):
        runner = _make_runner()
        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 5}}), \
             patch("tools.delegate_tool.list_active_subagents", return_value=[]), \
             patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_swarm_command(_make_event(text="/swarm stop the bleeding in auth"))

        assert "Usage: /swarm stop <id>" in result
        assert "use `/swarm start <prompt>`" in result
        assert len(created_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_allows_prompt_beginning_with_stop_word(self):
        runner = _make_runner()
        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 5}}), \
             patch("tools.delegate_tool.list_active_subagents", return_value=[]), \
             patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_swarm_command(_make_event(text="/swarm start stop the bleeding in auth"))

        assert "Detached swarm started" in result
        assert len(created_tasks) == 1

    @pytest.mark.asyncio
    async def test_launch_rejected_when_chat_already_has_active_foreman(self):
        runner = _make_runner()
        rows = [{
            "subagent_id": "swarm_busy",
            "kind": "detached_foreman",
            "source_platform": "telegram",
            "source_chat_id": "67890",
            "source_thread_id": "",
        }]
        with patch("tools.delegate_tool.list_active_subagents", return_value=rows):
            result = await runner._handle_swarm_command(_make_event(text="/swarm investigate auth"))
        assert "already active for this chat" in result

    @pytest.mark.asyncio
    async def test_stop_with_multiple_matching_foremen_requires_explicit_id(self):
        runner = _make_runner()
        rows = [
            {
                "subagent_id": "swarm_1",
                "kind": "detached_foreman",
                "source_platform": "telegram",
                "source_chat_id": "67890",
                "source_thread_id": "",
            },
            {
                "subagent_id": "swarm_2",
                "kind": "detached_foreman",
                "source_platform": "telegram",
                "source_chat_id": "67890",
                "source_thread_id": "",
            },
        ]
        with patch("tools.delegate_tool.list_active_subagents", return_value=rows):
            result = await runner._handle_swarm_command(_make_event(text="/swarm stop"))
        assert "Multiple detached swarms" in result

    @pytest.mark.asyncio
    async def test_swarm_status_labels_detached_foreman_rows(self):
        runner = _make_runner()
        with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 6}}), \
             patch("tools.delegate_tool.is_spawn_paused", return_value=False), \
             patch("tools.delegate_tool._get_max_spawn_depth", return_value=1), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=3), \
             patch("tools.delegate_tool.list_active_subagents", return_value=[{
                 "subagent_id": "swarm_123",
                 "kind": "detached_foreman",
                 "depth": 0,
                 "status": "running",
                 "stalled": False,
                 "idle_seconds": 12.0,
                 "model": "gpt-5.4",
                 "goal": "Orchestrate worker swarm",
                 "source_platform": "telegram",
                 "source_chat_id": "67890",
                 "source_thread_id": "",
             }]):
            result = await runner._handle_swarm_command(_make_event(text="/swarm status"))
        assert "foreman" in result
        assert "Detached foreman worker cap:" in result
        assert "Active swarm agents in this chat:" in result

    @pytest.mark.asyncio
    async def test_swarm_status_is_scoped_to_this_chat_and_descendants(self):
        runner = _make_runner()
        rows = [
            {
                "subagent_id": "swarm_here",
                "kind": "detached_foreman",
                "depth": 0,
                "status": "running",
                "stalled": False,
                "idle_seconds": 5.0,
                "model": "gpt-5.4",
                "goal": "This chat goal",
                "source_platform": "telegram",
                "source_chat_id": "67890",
                "source_thread_id": "",
            },
            {
                "subagent_id": "sa-child",
                "parent_id": "swarm_here",
                "kind": "delegate_child",
                "depth": 1,
                "status": "running",
                "stalled": False,
                "idle_seconds": 2.0,
                "model": "gpt-5.4-mini",
                "goal": "Child worker goal",
            },
            {
                "subagent_id": "swarm_elsewhere",
                "kind": "detached_foreman",
                "depth": 0,
                "status": "running",
                "stalled": False,
                "idle_seconds": 8.0,
                "model": "gpt-5.4",
                "goal": "Other chat goal",
                "source_platform": "telegram",
                "source_chat_id": "different-chat",
                "source_thread_id": "",
            },
        ]
        with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 6}}), \
             patch("tools.delegate_tool.is_spawn_paused", return_value=False), \
             patch("tools.delegate_tool._get_max_spawn_depth", return_value=1), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=3), \
             patch("tools.delegate_tool.list_active_subagents", return_value=rows):
            result = await runner._handle_swarm_command(_make_event(text="/swarm status"))
        assert "swarm_here" in result
        assert "sa-child" in result
        assert "Child worker goal" in result
        assert "swarm_elsewhere" not in result
        assert "Other chat goal" not in result
        assert "**Other active swarm agents:** 1" in result

    @pytest.mark.asyncio
    async def test_swarm_pause_is_denied_outside_home_channel(self):
        runner = _make_runner()
        result = await runner._handle_swarm_command(_make_event(text="/swarm pause", chat_id="not-home"))
        assert "restricted to the platform home channel" in result

    @pytest.mark.asyncio
    async def test_swarm_resume_is_denied_outside_home_channel(self):
        runner = _make_runner()
        result = await runner._handle_swarm_command(_make_event(text="/swarm resume", chat_id="not-home"))
        assert "restricted to the platform home channel" in result

    @pytest.mark.asyncio
    async def test_run_swarm_task_registers_interruptible_foreman_and_filters_toolsets(self):
        runner = _make_runner()
        source = _make_event(text="/swarm test").source
        mock_adapter = MagicMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.send_image = AsyncMock()
        mock_adapter.send_document = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Swarm done"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Swarm done"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        runner._resolve_session_agent_runtime = MagicMock(return_value=("gpt-5.4", {"api_key": "key"}))
        runner._load_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(return_value={
            "model": "gpt-5.4",
            "runtime": {"api_key": "key"},
            "request_overrides": None,
        })
        runner._cleanup_agent_resources = MagicMock()

        captured = {}
        release = threading.Event()

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.interrupt = MagicMock()
            def run_conversation(self, user_message=None, task_id=None):
                release.wait(timeout=2)
                return {"final_response": "Swarm done"}

        async def delayed_executor(fn):
            return await asyncio.to_thread(fn)

        runner._run_in_executor_with_context = AsyncMock(side_effect=delayed_executor)

        with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 6}}), \
             patch("hermes_cli.tools_config._get_platform_tools", return_value={"terminal", "messaging", "memory"}), \
             patch("run_agent.AIAgent", FakeAgent):
            task = asyncio.create_task(runner._run_swarm_task("Investigate auth", source, "swarm_test"))
            await asyncio.sleep(0.05)
            from tools.delegate_tool import interrupt_subagent
            assert interrupt_subagent("swarm_test") is True
            release.set()
            await task

        assert "delegation" in captured["enabled_toolsets"]
        assert "terminal" in captured["enabled_toolsets"]
        assert "skills" in captured["enabled_toolsets"]
        assert "messaging" not in captured["enabled_toolsets"]
        assert "memory" not in captured["enabled_toolsets"]
        assert "messaging" in captured["disabled_toolsets"]
        assert "memory" in captured["disabled_toolsets"]
        mock_adapter.send.assert_called()
        sent_content = mock_adapter.send.call_args.kwargs.get("content", "")
        assert "Swarm task complete" in sent_content

    @pytest.mark.asyncio
    async def test_run_swarm_task_unregisters_launching_foreman_when_adapter_missing(self):
        runner = _make_runner()
        source = _make_event(text="/swarm test").source

        from tools.delegate_tool import _register_subagent, _unregister_subagent, list_active_subagents

        task_id = "swarm_missing_adapter"
        _register_subagent(
            {
                "subagent_id": task_id,
                "parent_id": None,
                "depth": 0,
                "goal": "Investigate auth",
                "status": "launching",
                "kind": "detached_foreman",
                "source_platform": source.platform.value,
                "source_chat_id": source.chat_id,
                "source_thread_id": source.thread_id,
            }
        )

        try:
            await runner._run_swarm_task("Investigate auth", source, task_id)
            assert not any(row["subagent_id"] == task_id for row in list_active_subagents())
        finally:
            _unregister_subagent(task_id)

    @pytest.mark.asyncio
    async def test_run_swarm_task_unregisters_launching_foreman_when_api_key_missing(self):
        runner = _make_runner()
        source = _make_event(text="/swarm test").source
        mock_adapter = MagicMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        runner._resolve_session_agent_runtime = MagicMock(return_value=("gpt-5.4", {}))

        from tools.delegate_tool import _register_subagent, _unregister_subagent, list_active_subagents

        task_id = "swarm_missing_api_key"
        _register_subagent(
            {
                "subagent_id": task_id,
                "parent_id": None,
                "depth": 0,
                "goal": "Investigate auth",
                "status": "launching",
                "kind": "detached_foreman",
                "source_platform": source.platform.value,
                "source_chat_id": source.chat_id,
                "source_thread_id": source.thread_id,
            }
        )

        try:
            with patch("gateway.run._load_gateway_config", return_value={"swarm": {"max_workers": 6}}):
                await runner._run_swarm_task("Investigate auth", source, task_id)

            mock_adapter.send.assert_awaited_once()
            assert not any(row["subagent_id"] == task_id for row in list_active_subagents())
        finally:
            _unregister_subagent(task_id)
