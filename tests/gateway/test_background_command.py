"""Tests for /bg gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

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
    async def test_successful_task_sends_result(self):
        """When the agent completes successfully, the result is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Hello from background!"))
        mock_adapter.toolsets_for_source = MagicMock(return_value=None)
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

    @pytest.mark.asyncio
    async def test_matrix_task_registers_isolated_interactive_approval(self, monkeypatch):
        """A dangerous background action must emit a card in its source room."""
        from gateway import run as gateway_run
        from tools import approval

        runner = _make_runner()
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("test-model", {"api_key": "test-key"})
        )
        runner._resolve_session_reasoning_config = MagicMock(return_value=None)
        runner._load_service_tier = MagicMock(return_value=None)
        runner._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": "test-model",
                "runtime": {"api_key": "test-key"},
                "request_overrides": None,
            }
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

        approvals = []
        typed_resolution_counts = []
        typed_session_key = ""

        class MatrixAdapter:
            typed_command_prefix = "!"

            async def send_exec_approval(self, **kwargs):
                approvals.append(kwargs)
                count = approval.resolve_gateway_approval(
                    typed_session_key,
                    "once",
                    approval_id=kwargs["metadata"]["approval_id"],
                )
                typed_resolution_counts.append(count)
                if not count:
                    approval.resolve_gateway_approval(
                        kwargs["session_key"],
                        "once",
                        approval_id=kwargs["metadata"]["approval_id"],
                    )
                return MagicMock(success=True, error=None)

            async def send(self, **_kwargs):
                return MagicMock(success=True, error=None)

            @staticmethod
            def extract_media(response):
                return [], response

            @staticmethod
            def extract_images(response):
                return [], response

        adapter = MatrixAdapter()
        runner.adapters[Platform.MATRIX] = adapter
        source = SessionSource(
            platform=Platform.MATRIX,
            user_id="@user:example.org",
            chat_id="!room:example.org",
            user_name="user-example",
            chat_type="group",
            thread_id="$thread",
        )
        task_id = "bg_approval_test"
        typed_session_key = runner._session_key_for_source(source)
        observed = {}

        monkeypatch.setattr(approval, "is_approved", lambda *_args: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)

        def run_conversation(**_kwargs):
            observed["session_key"] = approval.get_current_session_key(default="")
            observed["decision"] = approval.check_dangerous_command(
                "rm -rf -- /tmp/probe",
                env_type="local",
                has_host_access=True,
            )
            return {"final_response": "approved", "messages": []}

        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = run_conversation
        mock_agent.shutdown_memory_provider = MagicMock()
        mock_agent.close = MagicMock()

        try:
            with patch("run_agent.AIAgent", return_value=mock_agent):
                await runner._run_background_task(
                    "delete the probe",
                    source,
                    task_id,
                    event_message_id="$request",
                )
        finally:
            approval.clear_session(task_id)

        assert observed["session_key"] == task_id
        assert observed["decision"]["approved"] is True
        assert typed_resolution_counts == [1]
        assert len(approvals) == 1
        assert approvals[0]["chat_id"] == source.chat_id
        assert approvals[0]["command"] == "rm -rf -- /tmp/probe"
        assert approvals[0]["session_key"] == task_id
        assert approvals[0]["metadata"]["approval_id"]
        assert approvals[0]["metadata"]["requester_user_id"] == source.user_id
        assert approvals[0]["metadata"]["thread_id"] == source.thread_id
        with approval._lock:
            assert task_id not in approval._gateway_notify_cbs
            assert task_id not in approval._gateway_command_session_keys


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
