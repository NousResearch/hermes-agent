"""Tests for /bg gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

def _make_event(text="/bg", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)

def _make_runner():
    """Create a bare GatewayRunner with minimal mocks."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()

    mock_store = MagicMock()
    # A real SessionStore returns None when no persisted /model override exists.
    # MagicMock's default truthy return would otherwise rehydrate a fake model
    # and make the session-scoped reasoning resolver receive a MagicMock.
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner

# ---------------------------------------------------------------------------
# _handle_background_command
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------

class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""

    @pytest.mark.asyncio
    async def test_delegated_result_is_delivered_before_background_task_ends(self, monkeypatch):
        """A /bg session has no consumer for a detached completion (#140569)."""
        import tools.delegate_tool as dt
        from gateway.session_context import async_delivery_supported, clear_session_vars, set_session_vars

        runner = _make_runner()
        adapter = AsyncMock()
        adapter.toolsets_for_source = MagicMock(return_value=None)
        adapter.extract_media = lambda text: ([], text)
        adapter.extract_images = lambda text: ([], text)
        runner.adapters[Platform.TELEGRAM] = adapter
        parent = SimpleNamespace(_delegate_depth=0, _subagent_id=None)
        monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: SimpleNamespace(_subagent_id="child"))
        monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **kw: {
            "model": "test", "provider": None, "base_url": None, "api_key": None,
            "api_mode": None, "command": None, "args": None,
        })
        monkeypatch.setattr(dt, "_run_single_child", lambda **kw: {
            "task_index": 0, "status": "completed", "summary": "child work product",
            "api_calls": 1, "duration_seconds": 0, "model": "test", "exit_reason": "completed",
        })
        detached = MagicMock(return_value={"delegation_id": "detached"})
        monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation_batch", detached)
        agent = MagicMock()
        agent.run_conversation.side_effect = lambda **kw: {
            "final_response": dt.delegate_task(goal="inspect workspace", background=True, parent_agent=parent),
            "messages": [],
        }
        tokens = set_session_vars(platform="telegram", chat_id="67890", async_delivery=True)
        try:
            with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
                 patch("gateway.run._load_gateway_config", return_value={}), \
                 patch("run_agent.AIAgent", return_value=agent):
                await runner._run_background_task("inspect workspace", _make_event().source, "bg_test")
            assert async_delivery_supported() is True
        finally:
            clear_session_vars(tokens)

        detached.assert_not_called()
        assert "child work product" in json.dumps(adapter.send.call_args.kwargs)
        agent.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_task_sends_result(self):
        """When the agent completes successfully, the result is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello from background!"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        mock_result = {"final_response": "Hello from background!", "messages": []}

        checkpoint_config = {
            "checkpoints": {
                "enabled": True,
                "max_snapshots": 8,
                "max_total_size_mb": 222,
                "max_file_size_mb": 3,
            }
        }
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value=checkpoint_config), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        # Should have sent the result
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Hello from background!" in content
        agent_kwargs = MockAgent.call_args.kwargs
        assert agent_kwargs["checkpoints_enabled"] is True
        assert agent_kwargs["checkpoint_max_snapshots"] == 8
        assert agent_kwargs["checkpoint_max_total_size_mb"] == 222
        assert agent_kwargs["checkpoint_max_file_size_mb"] == 3
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

# ---------------------------------------------------------------------------
# /bg in help and known_commands
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI /bg command definition
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _handle_btw_command
# ---------------------------------------------------------------------------

class TestHandleBtwCommand:
    """Tests for GatewayRunner._handle_btw_command (context-aware side question)."""

    @pytest.mark.asyncio
    async def test_dispatches_side_question_and_sends_answer(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = [
            {"role": "user", "content": "fix foo.py"},
            {"role": "assistant", "content": "done"},
        ]
        store._store = runner.session_store
        runner._async_session_store = store
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "k", "provider": "p",
                                         "base_url": "u", "api_mode": "chat_completions"})
        )
        runner._reply_anchor_for_event = MagicMock(return_value=None)
        runner._thread_metadata_for_source = MagicMock(return_value=None)
        mock_adapter = AsyncMock()
        runner._delivery_adapter_for = MagicMock(return_value=mock_adapter)

        event = _make_event(text="/btw which file was that?")

        with patch("agent.side_question.answer_side_question",
                   return_value="it was foo.py") as mock_answer:
            result = await runner._handle_btw_command(event)
            # Ack returned immediately, worker task registered.
            assert "which file was that?" in result
            # Drain the fire-and-forget task.
            for task in list(runner._background_tasks):
                await task

        # Snapshot + question reached the engine; live history untouched.
        args, kwargs = mock_answer.call_args
        assert args[0] == "which file was that?"
        assert args[1][0]["content"] == "fix foo.py"
        assert kwargs["main_runtime"]["model"] == "test-model"

        # The answer was delivered to the chat.
        mock_adapter.send.assert_called_once()
        sent_text = mock_adapter.send.call_args[0][1]
        assert "it was foo.py" in sent_text
