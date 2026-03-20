"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/background", platform=Platform.TELEGRAM,
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

    mock_store = MagicMock()
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
        """Running /background with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result
        assert "/background" in result

    @pytest.mark.asyncio
    async def test_bg_alias_no_prompt_shows_usage(self):
        """Running /bg with no prompt shows usage."""
        runner = _make_runner()
        event = _make_event(text="/bg")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_shows_usage(self):
        """Running /background with only whitespace shows usage."""
        runner = _make_runner()
        event = _make_event(text="/background   ")
        result = await runner._handle_background_command(event)
        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_valid_prompt_starts_task(self):
        """Running /background with a prompt returns confirmation and starts task."""
        runner = _make_runner()

        # Patch asyncio.create_task to capture the coroutine
        created_tasks = []
        original_create_task = asyncio.create_task

        def capture_task(coro, *args, **kwargs):
            # Close the coroutine to avoid warnings
            coro.close()
            mock_task = MagicMock()
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/background Summarize the top HN stories")
            result = await runner._handle_background_command(event)

        assert "🔄" in result
        assert "Background task started" in result
        assert "bg_" in result  # task ID starts with bg_
        assert "Summarize the top HN stories" in result
        assert len(created_tasks) == 1  # background task was created

    @pytest.mark.asyncio
    async def test_prompt_truncated_in_preview(self):
        """Long prompts are truncated to 60 chars in the confirmation message."""
        runner = _make_runner()
        long_prompt = "A" * 100

        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            event = _make_event(text=f"/background {long_prompt}")
            result = await runner._handle_background_command(event)

        assert "..." in result
        # Should not contain the full prompt
        assert long_prompt not in result

    @pytest.mark.asyncio
    async def test_task_id_is_unique(self):
        """Each background task gets a unique task ID."""
        runner = _make_runner()
        task_ids = set()

        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            for i in range(5):
                event = _make_event(text=f"/background task {i}")
                result = await runner._handle_background_command(event)
                # Extract task ID from result (format: "Task ID: bg_HHMMSS_hex")
                for line in result.split("\n"):
                    if "Task ID:" in line:
                        tid = line.split("Task ID:")[1].strip()
                        task_ids.add(tid)

        assert len(task_ids) == 5  # all unique

    @pytest.mark.asyncio
    async def test_works_across_platforms(self):
        """The /background command works for all platforms."""
        for platform in [Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK]:
            runner = _make_runner()
            with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
                event = _make_event(
                    text="/background test task",
                    platform=platform,
                )
                result = await runner._handle_background_command(event)
                assert "Background task started" in result


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    """Tests for GatewayRunner._run_background_task (the actual execution)."""

    @pytest.mark.asyncio
    async def test_no_adapter_returns_silently(self):
        """When no adapter is available, the task returns without error."""
        runner = _make_runner()
        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )
        # No adapters set — should not raise
        await runner._run_background_task("test prompt", source, "bg_test")

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

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        # Should have sent the result
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "Background task complete" in content
        assert "Hello from background!" in content

    @pytest.mark.asyncio
    async def test_exception_sends_error_message(self):
        """When the agent raises an exception, an error message is sent."""
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

        with patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=RuntimeError("boom")):
            await runner._run_background_task("test prompt", source, "bg_test")

        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "failed" in content.lower()


# ---------------------------------------------------------------------------
# /background in help and known_commands
# ---------------------------------------------------------------------------


class TestBackgroundInHelp:
    """Verify /background appears in help text and known commands."""

    @pytest.mark.asyncio
    async def test_background_in_help_output(self):
        """The /help output includes /background."""
        runner = _make_runner()
        event = _make_event(text="/help")
        result = await runner._handle_help_command(event)
        assert "/background" in result

    def test_background_is_known_command(self):
        """The /background command is in GATEWAY_KNOWN_COMMANDS."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "background" in GATEWAY_KNOWN_COMMANDS

    def test_bg_alias_is_known_command(self):
        """The /bg alias is in GATEWAY_KNOWN_COMMANDS."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "bg" in GATEWAY_KNOWN_COMMANDS


# ---------------------------------------------------------------------------
# CLI /background command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /background is registered in the CLI command system."""

    def test_background_in_commands_dict(self):
        """The /background command is in the COMMANDS dict."""
        from hermes_cli.commands import COMMANDS
        assert "/background" in COMMANDS

    def test_bg_alias_in_commands_dict(self):
        """The /bg alias is in the COMMANDS dict."""
        from hermes_cli.commands import COMMANDS
        assert "/bg" in COMMANDS

    def test_background_in_session_category(self):
        """The /background command is in the Session category."""
        from hermes_cli.commands import COMMANDS_BY_CATEGORY
        assert "/background" in COMMANDS_BY_CATEGORY["Session"]

    def test_background_autocompletes(self):
        """The /background command appears in autocomplete results."""
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("backgro")  # Partial match
        completions = list(completer.get_completions(doc, None))
        # Text doesn't start with / so no completions
        assert len(completions) == 0

        doc = Document("/backgro")  # With slash prefix
        completions = list(completer.get_completions(doc, None))
        cmd_displays = [str(c.display) for c in completions]
        assert any("/background" in d for d in cmd_displays)
