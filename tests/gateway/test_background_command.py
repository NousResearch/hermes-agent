"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
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
        content = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert content == (
            "結果：背景任務未執行\n"
            "變更：目前無法安全啟動背景任務。\n"
            "下一步：請檢查模型供應商設定後重試。"
        )

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
        assert content == "Hello from background!"
        agent_kwargs = MockAgent.call_args.kwargs
        assert agent_kwargs["checkpoints_enabled"] is True
        assert agent_kwargs["checkpoint_max_snapshots"] == 8
        assert agent_kwargs["checkpoint_max_total_size_mb"] == 222
        assert agent_kwargs["checkpoint_max_file_size_mb"] == 3
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_telegram_result_error_omits_internal_error_details(self, monkeypatch):
        from gateway import run as gateway_run

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
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "", "error": "internal-sensitive-detail"}
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(side_effect=lambda text: ([], text))
        mock_adapter.extract_images = MagicMock(side_effect=lambda text: ([], text))
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890")

        await runner._run_background_task("private prompt", source, "bg_test")

        content = mock_adapter.send.call_args.kwargs["content"]
        assert content == (
            "結果：背景任務失敗\n"
            "變更：背景處理未能安全完成。\n"
            "下一步：請重試；若持續失敗，再檢查 Gateway 日誌。"
        )
        assert "internal-sensitive-detail" not in content

    @pytest.mark.asyncio
    async def test_telegram_empty_extracted_response_is_canonical(self, monkeypatch):
        from gateway import run as gateway_run

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
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "removed during extraction", "messages": []}
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], ""))
        mock_adapter.extract_images = MagicMock(return_value=([], ""))
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890")

        await runner._run_background_task("private prompt", source, "bg_test")

        assert mock_adapter.send.call_args.kwargs["content"] == (
            "結果：背景任務已完成\n"
            "變更：任務沒有產生可顯示的文字或附件。\n"
            "下一步：如有需要，請調整要求後重試。"
        )

    @pytest.mark.asyncio
    async def test_telegram_exception_delivery_omits_exception_and_task_details(self, monkeypatch):
        from gateway import run as gateway_run

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
        runner._run_in_executor_with_context = AsyncMock(
            side_effect=RuntimeError("credential-shaped-internal-detail")
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter
        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
        )

        await runner._run_background_task("private prompt", source, "bg_secret_task")

        content = mock_adapter.send.call_args.kwargs["content"]
        assert content == (
            "結果：背景任務失敗\n"
            "變更：背景處理未能安全完成。\n"
            "下一步：請重試；若持續失敗，再檢查 Gateway 日誌。"
        )
        assert "credential-shaped" not in content
        assert "private prompt" not in content
        assert "bg_secret_task" not in content

    @pytest.mark.asyncio
    async def test_media_files_routed_by_type(self, monkeypatch):
        """Result media is routed to the type-specific sender, not send_document.

        A TTS clip should arrive as a voice bubble, a video as a video, an
        image as a native image, and everything else as a document.
        """
        from gateway import run as gateway_run

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
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "see attached", "messages": []}
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

        # Four real files so the media-delivery path validator accepts them
        # (default mode requires the file to exist as a regular file).
        import os as _os
        import tempfile as _tempfile
        _tmpdir = _tempfile.mkdtemp(prefix="bg_media_")
        _ogg = _os.path.join(_tmpdir, "clip.ogg")
        _mp4 = _os.path.join(_tmpdir, "render.mp4")
        _png = _os.path.join(_tmpdir, "chart.png")
        _pdf = _os.path.join(_tmpdir, "report.pdf")
        for _p in (_ogg, _mp4, _png, _pdf):
            with open(_p, "wb") as _fh:
                _fh.write(b"x")
        # ogg flagged as voice, mp4 video, png image, pdf doc.
        media = [
            (_ogg, True),
            (_mp4, False),
            (_png, False),
            (_pdf, False),
        ]

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        mock_adapter.send_video = AsyncMock()
        mock_adapter.send_image_file = AsyncMock()
        mock_adapter.send_document = AsyncMock()
        mock_adapter.send_image = AsyncMock()
        # No text, no markdown images — just the four media attachments.
        mock_adapter.extract_media = MagicMock(return_value=(media, ""))
        mock_adapter.extract_images = MagicMock(return_value=([], ""))
        # Non-telegram platform so every audio ext routes through send_voice.
        runner.adapters[Platform.DISCORD] = mock_adapter

        source = SessionSource(
            platform=Platform.DISCORD,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        try:
            await runner._run_background_task("make stuff", source, "bg_test")

            mock_adapter.send_voice.assert_called_once()
            assert mock_adapter.send_voice.call_args.kwargs["audio_path"] == _ogg
            mock_adapter.send_video.assert_called_once()
            assert mock_adapter.send_video.call_args.kwargs["video_path"] == _mp4
            mock_adapter.send_image_file.assert_called_once()
            assert mock_adapter.send_image_file.call_args.kwargs["image_path"] == _png
            mock_adapter.send_document.assert_called_once()
            assert mock_adapter.send_document.call_args.kwargs["file_path"] == _pdf
        finally:
            import shutil as _shutil
            _shutil.rmtree(_tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_telegram_dm_topic_completion_preserves_reply_anchor_metadata(self, monkeypatch):
        """Background completion metadata must let Telegram send thread id plus reply id."""
        from gateway import run as gateway_run

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
        runner._run_in_executor_with_context = AsyncMock(
            return_value={"final_response": "done", "messages": []}
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "done"))
        mock_adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="20197",
        )

        await runner._run_background_task(
            "say hello",
            source,
            "bg_test",
            event_message_id="463",
        )

        mock_adapter.send.assert_called_once()
        assert mock_adapter.send.call_args.kwargs["metadata"] == {
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "20197",
            "telegram_reply_to_message_id": "463",
        }

    @pytest.mark.asyncio
    async def test_agent_cleanup_runs_when_background_agent_raises(self):
        """Temporary background agents must be cleaned up on error paths too."""
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

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.shutdown_memory_provider = MagicMock()
            mock_agent_instance.close = MagicMock()
            mock_agent_instance.run_conversation.side_effect = RuntimeError("boom")
            MockAgent.return_value = mock_agent_instance

            await runner._run_background_task("say hello", source, "bg_test")

        mock_adapter.send.assert_called_once()
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

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
        assert content == (
            "結果：背景任務失敗\n"
            "變更：背景處理未能安全完成。\n"
            "下一步：請重試；若持續失敗，再檢查 Gateway 日誌。"
        )
        assert "boom" not in content

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
