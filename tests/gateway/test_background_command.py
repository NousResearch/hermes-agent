"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, EphemeralReply, MessageEvent, SendResult
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

    @pytest.mark.asyncio
    async def test_ack_is_ephemeral_so_it_is_never_scanned_for_attachments(self):
        """The ack must not be scanned by extract_local_files() at all.

        Regression for #64661: the ack text embeds a truncated echo of the
        user's prompt. If that prompt contains a bare existing file path,
        extract_local_files() (run by _process_message_background on every
        *non-ephemeral* response, per gateway/platforms/base.py) would upload
        it from the ack — and the background task's own intentional delivery
        of the same file then uploads it a second time. Returning an
        EphemeralReply (the same mechanism /stop, /restart, /reset and /yolo
        already use for status notices) makes _process_message_background
        skip extraction for this reply entirely, regardless of what the
        echoed preview contains.
        """
        runner = _make_runner()
        prompt = "use /tmp/bg.png and return it as an image"
        with patch("gateway.run.asyncio.create_task", side_effect=lambda c, **kw: (c.close(), MagicMock())[1]):
            event = _make_event(text=f"/background {prompt}")
            result = await runner._handle_background_command(event)

        assert isinstance(result, EphemeralReply)
        assert result.ttl_seconds == 0  # no auto-delete — prior behavior preserved

        # Sanity: the path really would be picked up if this weren't skipped
        # via the is_ephemeral_response guard (confirms the test is not
        # trivially passing because the path can never match).
        with patch("os.path.isfile", return_value=True):
            local_files, _ = BasePlatformAdapter.extract_local_files(str(result))
        assert local_files == ["/tmp/bg.png"]

    @pytest.mark.asyncio
    async def test_background_ack_and_completion_deliver_file_exactly_once(self, tmp_path):
        """Full adapter-pipeline regression for #64661.

        Runs the ack and the background task's completion notice through the
        real `_process_message_background` pipeline (not just the standalone
        extractor, per the maintainer review on this PR) with a real existing
        file, and asserts the required delivery invariant directly: the
        acknowledgement never triggers an attachment, and the completion
        notice triggers the file's *only* upload.
        """
        result_path = tmp_path / "bg_result.txt"
        result_path.write_text("background task output", encoding="utf-8")

        class _Adapter(BasePlatformAdapter):
            async def connect(self, *, is_reconnect: bool = False):
                pass

            async def disconnect(self):
                pass

            async def send(self, chat_id, content="", **kwargs):
                return SendResult(success=True, message_id="m-1")

            async def get_chat_info(self, chat_id):
                return {}

        adapter = _Adapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
        adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="sent-1"))
        adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc-1"))

        session_key = "agent:main:telegram:private:67890"
        ack_event = _make_event(text=f"/background summarize {result_path}")
        completion_event = _make_event(text="/background")  # source only; text is the prior command

        # Step 1 — the ack, exactly as _handle_background_command returns it:
        # an EphemeralReply whose truncated echo can contain the same path.
        async def _ack_handler(evt):
            return EphemeralReply(f"⏳ Running in background... (prompt: summarize {result_path})", ttl_seconds=0)

        adapter.set_message_handler(_ack_handler)
        with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()), \
             patch.object(adapter, "_keep_typing", new=AsyncMock()):
            await adapter._process_message_background(ack_event, session_key)

        adapter.send_document.assert_not_awaited()

        # Step 2 — the background task's own completion delivery: a plain
        # response naming the same file, exactly as _run_background_task
        # sends it once the task finishes.
        async def _completion_handler(evt):
            return f"Done! Output saved to {result_path}"

        adapter.set_message_handler(_completion_handler)
        with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()), \
             patch.object(adapter, "_keep_typing", new=AsyncMock()):
            await adapter._process_message_background(completion_event, session_key)

        adapter.send_document.assert_awaited_once()
        assert adapter.send_document.call_args.kwargs["file_path"] == str(result_path)



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
