"""Tests for /bg gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
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


class TestHandleBackgroundCommand:
    """Tests for GatewayRunner._handle_background_command."""

    @pytest.mark.asyncio
    async def test_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/bg" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /bg with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""


    @pytest.mark.asyncio
    async def test_no_credentials_sends_error(self):
        """When provider credentials are missing, an error is sent."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("test prompt", source, "bg_test")

        # Should have sent an error message
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert "failed" in call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "").lower()

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
        assert "Background task complete" in content
        assert "Hello from background!" in content
        agent_kwargs = MockAgent.call_args.kwargs
        assert agent_kwargs["checkpoints_enabled"] is True
        assert agent_kwargs["checkpoint_max_snapshots"] == 8
        assert agent_kwargs["checkpoint_max_total_size_mb"] == 222
        assert agent_kwargs["checkpoint_max_file_size_mb"] == 3
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# _run_background_task inactivity watchdog + cancellation (PR #8298)
# ---------------------------------------------------------------------------


def _sent_contents(mock_adapter):
    """Collect the text content of every adapter.send() call (positional or kw)."""
    out = []
    for call in mock_adapter.send.call_args_list:
        content = call.kwargs.get("content")
        if content is None and len(call.args) > 1:
            content = call.args[1]
        out.append(content or "")
    return out


def _prime_worker_runner(runner, monkeypatch):
    """Wire the resolve helpers so _run_background_task reaches run_sync (which
    builds the real agent via the patched AIAgent and populates agent_holder)."""
    from gateway import run as gateway_run

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
    runner._refresh_fallback_model = MagicMock(return_value=None)
    runner._cleanup_agent_resources = MagicMock()
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})


class TestBackgroundWatchdog:
    """The /background inactivity watchdog + cancellation handling."""

    @pytest.mark.asyncio
    async def test_inactivity_warning_fires_once(self, monkeypatch):
        """A single idle-warning message is sent while the agent stays idle,
        even though the poller ticks several times before the run completes."""
        import time as _time

        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "9999")   # never time out
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0.01")
        monkeypatch.setenv("HERMES_BG_POLL_INTERVAL", "0.02")

        runner = _make_runner()
        _prime_worker_runner(runner, monkeypatch)

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "done late"))
        mock_adapter.extract_images = MagicMock(return_value=([], "done late"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="12345", chat_id="67890",
            user_name="testuser",
        )

        mock_agent = MagicMock()
        mock_agent.get_activity_summary.return_value = {
            "seconds_since_activity": 999.0, "current_tool": "shell",
        }
        mock_agent.run_conversation.side_effect = lambda **kw: (
            _time.sleep(0.12) or {"final_response": "done late", "messages": []}
        )
        mock_agent.shutdown_memory_provider = MagicMock()
        mock_agent.close = MagicMock()

        with patch("run_agent.AIAgent", return_value=mock_agent):
            await runner._run_background_task("do slow thing", source, "bg_test")

        contents = _sent_contents(mock_adapter)
        warnings = [c for c in contents if "no activity for" in c and "time out soon" in c]
        completions = [c for c in contents if "Background task complete" in c]
        assert len(warnings) == 1, f"expected exactly one idle warning, got {contents}"
        assert len(completions) == 1, f"expected the result to still be delivered, got {contents}"

    @pytest.mark.asyncio
    async def test_inactivity_timeout_interrupts_and_notifies(self, monkeypatch):
        """When idle exceeds the timeout, the agent is interrupted and a
        timeout diagnostic is delivered (instead of hanging forever)."""
        import threading

        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0.05")
        monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0.01")
        monkeypatch.setenv("HERMES_BG_POLL_INTERVAL", "0.02")

        runner = _make_runner()
        _prime_worker_runner(runner, monkeypatch)

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="12345", chat_id="67890",
            user_name="testuser",
        )

        released = threading.Event()
        mock_agent = MagicMock()
        mock_agent.get_activity_summary.return_value = {
            "seconds_since_activity": 999.0, "current_tool": "shell",
        }
        # Block until interrupt() releases us (or a safety timeout), mimicking a
        # wedged worker thread that only unwinds once interrupted.
        mock_agent.run_conversation.side_effect = lambda **kw: (
            released.wait(timeout=5.0) and None
            or {"final_response": "late", "messages": []}
        )
        mock_agent.interrupt.side_effect = lambda *a, **k: released.set()
        mock_agent.shutdown_memory_provider = MagicMock()
        mock_agent.close = MagicMock()

        with patch("run_agent.AIAgent", return_value=mock_agent):
            await runner._run_background_task("wedge please", source, "bg_test")

        mock_agent.interrupt.assert_called()
        contents = _sent_contents(mock_adapter)
        assert any("timed out" in c for c in contents), contents
        assert any("`shell`" in c for c in contents), contents
        # Let the released worker thread unwind so its future is consumed cleanly.
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_cancellation_interrupts_and_notifies(self, monkeypatch):
        """Cancelling the background task (gateway shutdown) interrupts the
        in-flight agent and notifies the user rather than dying silently."""
        import threading

        monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "9999")   # not a timeout test
        monkeypatch.setenv("HERMES_BG_POLL_INTERVAL", "0.02")

        runner = _make_runner()
        _prime_worker_runner(runner, monkeypatch)

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM, user_id="12345", chat_id="67890",
            user_name="testuser",
        )

        started = threading.Event()
        released = threading.Event()
        mock_agent = MagicMock()
        mock_agent.get_activity_summary.return_value = {"seconds_since_activity": 0.0}
        mock_agent.run_conversation.side_effect = lambda **kw: (
            started.set() or released.wait(timeout=5.0)
            or {"final_response": "late", "messages": []}
        )
        mock_agent.interrupt.side_effect = lambda *a, **k: released.set()
        mock_agent.shutdown_memory_provider = MagicMock()
        mock_agent.close = MagicMock()

        with patch("run_agent.AIAgent", return_value=mock_agent):
            task = asyncio.ensure_future(
                runner._run_background_task("long job", source, "bg_test")
            )
            # Wait until the worker is actually running (agent_holder populated).
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set(), "worker never started"
            task.cancel()
            await task  # handler swallows CancelledError after notifying

        mock_agent.interrupt.assert_called()
        contents = _sent_contents(mock_adapter)
        assert any("was cancelled" in c for c in contents), contents


# ---------------------------------------------------------------------------
# /bg in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /bg and /btw appear in help text and known commands."""

    @pytest.mark.asyncio
    async def test_bg_and_btw_in_help_output(self):
        """The /help output includes /bg and /btw."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/bg" in result
        assert "/btw" in result


# ---------------------------------------------------------------------------
# CLI /bg command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /bg and /btw are registered in the CLI command system."""


    def test_bg_autocompletes(self):
        """The /bg and /btw commands appear in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("bg")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/bg")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/bg" in d for d in cmd_displays)

        doc = Document("/btw")
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/btw" in d for d in cmd_displays)


# ---------------------------------------------------------------------------
# _handle_btw_command
# ---------------------------------------------------------------------------


class TestHandleBtwCommand:
    """Tests for GatewayRunner._handle_btw_command (context-aware side question)."""

    @pytest.mark.asyncio
    async def test_no_question_shows_usage(self):
        runner = _make_runner()
        event = _make_event(text="/btw")
        result = await runner._handle_btw_command(event)
        assert "Usage:" in result
        assert "/btw" in result

    @pytest.mark.asyncio
    async def test_no_history_reports_no_conversation(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = []
        store._store = runner.session_store
        runner._async_session_store = store
        event = _make_event(text="/btw what did we do?")
        result = await runner._handle_btw_command(event)
        assert "conversation" in result.lower()

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
        runner._adapter_for_source = MagicMock(return_value=mock_adapter)

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

    @pytest.mark.asyncio
    async def test_no_credentials_reports_error(self):
        runner = _make_runner()
        store = AsyncMock()
        store.get_or_create_session.return_value = MagicMock(session_id="s1")
        store.load_transcript.return_value = [{"role": "user", "content": "hi"}]
        store._store = runner.session_store
        runner._async_session_store = store
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=(None, {"api_key": None})
        )
        event = _make_event(text="/btw what?")
        result = await runner._handle_btw_command(event)
        assert "❌" in result
