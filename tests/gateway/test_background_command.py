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

    @pytest.mark.asyncio
    async def test_recalled_memory_is_stripped_from_background_result(self):
        """Provider-echoed context must not reach /background's SMTP body."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        runner = _make_runner()
        raw_response = (
            "<memory-context>\nPRIVATE_SENTINEL_81312_BACKGROUND\n"
            "</memory-context>\nVisible background result"
        )
        with patch.dict(
            os.environ,
            {
                "EMAIL_ADDRESS": "hermes@test.com",
                "EMAIL_PASSWORD": "secret",
                "EMAIL_IMAP_HOST": "imap.test.com",
                "EMAIL_SMTP_HOST": "smtp.test.com",
            },
        ):
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        runner.adapters[Platform.EMAIL] = adapter
        source = SessionSource(
            platform=Platform.EMAIL,
            user_id="user@test.com",
            chat_id="user@test.com",
            user_name="testuser",
        )

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}
        ), patch("run_agent.AIAgent") as MockAgent, patch("smtplib.SMTP") as mock_smtp:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run_conversation.return_value = {
                "final_response": raw_response,
                "messages": [],
            }
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        sent_msg = mock_smtp.return_value.send_message.call_args.args[0]
        body = sent_msg.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "PRIVATE_SENTINEL_81312_BACKGROUND" not in body
        assert "Visible background result" in body

    @pytest.mark.asyncio
    async def test_background_exception_detail_is_fenced_before_smtp(self):
        """Provider exception details must not reach the human-facing SMTP body."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        runner = _make_runner()
        with patch.dict(
            os.environ,
            {
                "EMAIL_ADDRESS": "hermes@test.com",
                "EMAIL_PASSWORD": "secret",
                "EMAIL_IMAP_HOST": "imap.test.com",
                "EMAIL_SMTP_HOST": "smtp.test.com",
            },
        ):
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        runner.adapters[Platform.EMAIL] = adapter
        source = SessionSource(
            platform=Platform.EMAIL,
            user_id="user@test.com",
            chat_id="user@test.com",
            user_name="testuser",
        )
        private_context = "PRIVATE_SENTINEL_81312_BACKGROUND_EXCEPTION"
        inert_secret = "sk-INERT_81312_BACKGROUND_EXCEPTION"
        raw_detail = (
            f"{inert_secret}\n<memory-context>\n{private_context}\n</memory-context>"
            "\nprovider temporarily unavailable"
        )

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=RuntimeError(raw_detail),
        ), patch("smtplib.SMTP") as mock_smtp:
            await runner._run_background_task("say hello", source, "bg_test")

        sent_msg = mock_smtp.return_value.send_message.call_args.args[0]
        body = sent_msg.get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "Background task bg_test failed" in body
        assert "provider temporarily unavailable" in body
        assert private_context not in body
        assert inert_secret not in body
        assert "memory-context" not in body

    @pytest.mark.asyncio
    async def test_background_fully_fenced_exception_uses_generic_error(self):
        """An empty fenced detail falls back to a generic failure message."""
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
        raw_detail = "<memory-context>\nprivate exception detail\n</memory-context>"

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=RuntimeError(raw_detail),
        ):
            await runner._run_background_task("say hello", source, "bg_test")

        assert mock_adapter.send.call_args.kwargs["content"] == (
            "❌ Background task bg_test failed"
        )

    @pytest.mark.asyncio
    async def test_background_safe_exception_detail_is_preserved(self):
        """Benign exception context remains useful after the delivery fence."""
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

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            side_effect=RuntimeError("provider temporarily unavailable"),
        ):
            await runner._run_background_task("say hello", source, "bg_test")

        content = mock_adapter.send.call_args.kwargs["content"]
        assert content == (
            "❌ Background task bg_test failed: provider temporarily unavailable"
        )


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


# ---------------------------------------------------------------------------
# CLI /background command definition
# ---------------------------------------------------------------------------


class TestBackgroundInCLICommands:
    """Verify /background is registered in the CLI command system."""


    def test_background_autocompletes(self):
        """The /background command appears in autocomplete results."""
        pytest.importorskip("prompt_toolkit")
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
