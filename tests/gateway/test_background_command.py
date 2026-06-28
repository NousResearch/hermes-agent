"""Tests for /background gateway slash command.

Tests the _handle_background_command handler (run a prompt in a separate
background session) across gateway messenger platforms.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote

import pytest

from gateway.config import Platform
from gateway.config import GatewayConfig
from gateway.config import HomeChannel
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
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
    async def test_telegram_dm_topic_passes_trigger_anchor_to_task(self):
        """Telegram private-topic completion sends need the original command message id."""
        runner = _make_runner()
        runner._run_background_task = AsyncMock()

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            return mock_task

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            chat_type="dm",
            thread_id="20197",
        )
        event = MessageEvent(
            text="/background summarize",
            source=source,
            message_id="463",
            reply_to_message_id="462",
        )

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            result = await runner._handle_background_command(event)

        assert "Background task started" in result
        runner._run_background_task.assert_called_once()
        assert runner._run_background_task.call_args.kwargs["event_message_id"] == "463"

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


class TestCodexMediaRequestIntercept:
    """Tests for XiaoXing QQ media closed-loop requests."""

    def _napcat_runner(self, platform_name="napcat"):
        runner = _make_runner()
        napcat = Platform(platform_name)
        runner.config = GatewayConfig(
            platforms={
                napcat: PlatformConfig(
                    enabled=True,
                    home_channel=HomeChannel(
                        platform=napcat,
                        chat_id="490008192",
                        name="Dad QQ",
                    ),
                    extra={
                        "allowed_users": "490008192",
                        "group_allow_from": ["610066383"],
                    },
                )
            }
        )
        runner._start_background_task = AsyncMock(
            return_value="Background task started"
        )
        return runner, napcat

    @pytest.mark.asyncio
    async def test_dad_group_codex_image_request_starts_media_background_task(self):
        runner, napcat = self._napcat_runner()
        source = SessionSource(
            platform=napcat,
            chat_id="group:610066383",
            chat_type="group",
            user_id="490008192",
            user_name="李文浩",
        )
        event = MessageEvent(
            text="你再试试让 Codex 生成“骡子跳水”的图，记得发回来嗯。",
            source=source,
            message_id="m1",
        )

        result = await runner._maybe_handle_codex_media_request(event)

        assert result == "Background task started"
        runner._start_background_task.assert_awaited_once()
        call = runner._start_background_task.call_args
        prompt = call.args[0]
        assert "骡子跳水" in prompt
        assert "MEDIA:/absolute/path" in prompt
        assert "static-frame" in prompt
        assert "Updream direct API/script execution from Codex" in prompt
        assert "Do not use Hermes image_generate/image_gen" in prompt
        assert "XiaoXing is only the requester/channel bridge" in prompt
        assert "Do not call codex_image_request or codex_video_request" in prompt
        assert "/Users/heavenwistful/.codex/skills/updream-platform/scripts/updream_api.mjs" in prompt
        assert call.kwargs["extra_enabled_toolsets"] == ("terminal", "file")
        assert call.kwargs["excluded_toolsets"] == ("xiaoxing_codex_media", "image_gen")

    @pytest.mark.asyncio
    async def test_dad_dm_codex_image_request_starts_media_background_task(self):
        runner, napcat = self._napcat_runner()
        source = SessionSource(
            platform=napcat,
            chat_id="490008192",
            chat_type="dm",
            user_id="490008192",
            user_name="李文浩",
        )
        event = MessageEvent(
            text="让 Codex 生成一张小星看书的图发回来",
            source=source,
            message_id="m2",
        )

        result = await runner._maybe_handle_codex_media_request(event)

        assert result == "Background task started"
        runner._start_background_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dad_group_codex_image_request_can_route_to_mentor_channel(self):
        runner, napcat = self._napcat_runner()
        runner.config.platforms[napcat].extra["codex_media_route"] = "mentor_channel"
        runner.config.platforms[napcat].extra["codex_mentor_url"] = "http://127.0.0.1:8767/messages"
        runner._post_codex_mentor_task = AsyncMock(
            return_value={"ok": True, "queued": "background", "thread_id": "codex-thread-1"}
        )
        source = SessionSource(
            platform=napcat,
            chat_id="group:610066383",
            chat_type="group",
            user_id="490008192",
            user_name="李文浩",
        )
        event = MessageEvent(
            text="让 Codex 生成一张小星试麦克风的图，做好发回来",
            source=source,
            message_id="m-mentor-1",
        )

        result = await runner._maybe_handle_codex_media_request(event)

        assert "Codex mentor channel" in result
        runner._start_background_task.assert_not_awaited()
        runner._post_codex_mentor_task.assert_awaited_once()
        url, payload = runner._post_codex_mentor_task.call_args.args
        assert url == "http://127.0.0.1:8767/messages"
        assert payload["direction"] == "xiaoxing_to_codex"
        assert payload["metadata"]["request_type"] == "codex_media_generation"
        assert payload["metadata"]["media_kind"] == "image"
        assert payload["metadata"]["return_to"]["platform"] == "napcat"
        assert payload["metadata"]["return_to"]["chat_id"] == "group:610066383"
        assert payload["metadata"]["return_to"]["chat_type"] == "group"
        assert payload["metadata"]["return_to"]["user_id"] == "490008192"
        assert payload["metadata"]["return_to"]["message_id"] == "m-mentor-1"
        assert "小星试麦克风" in payload["body"]
        assert "Return target: napcat:group:610066383" in payload["body"]
        assert "MEDIA:/absolute/path" in payload["body"]

    @pytest.mark.asyncio
    async def test_milky_dad_group_codex_image_request_can_route_to_mentor_channel(self):
        runner, milky = self._napcat_runner("milky")
        runner.config.platforms[milky].extra["codex_media_route"] = "mentor_channel"
        runner.config.platforms[milky].extra["codex_mentor_url"] = "http://127.0.0.1:8767/messages"
        runner._post_codex_mentor_task = AsyncMock(
            return_value={"ok": True, "queued": "background", "thread_id": "codex-thread-2"}
        )
        source = SessionSource(
            platform=milky,
            chat_id="group:610066383",
            chat_type="group",
            user_id="490008192",
            user_name="李文浩",
        )
        event = MessageEvent(
            text="让 Codex 生成一张小星检查频道的图，做好发回来",
            source=source,
            message_id="m-milky-mentor-1",
        )

        result = await runner._maybe_handle_codex_media_request(event)

        assert "Codex mentor channel" in result
        runner._start_background_task.assert_not_awaited()
        runner._post_codex_mentor_task.assert_awaited_once()
        payload = runner._post_codex_mentor_task.call_args.args[1]
        assert payload["metadata"]["return_to"]["platform"] == "milky"
        assert payload["metadata"]["return_to"]["chat_id"] == "group:610066383"
        assert "Return target: milky:group:610066383" in payload["body"]
        assert "小星检查频道" in payload["body"]

    @pytest.mark.asyncio
    async def test_dad_group_codex_video_request_uses_updream_background_task(self):
        runner, napcat = self._napcat_runner()
        source = SessionSource(
            platform=napcat,
            chat_id="group:610066383",
            chat_type="group",
            user_id="490008192",
            user_name="李文浩",
        )
        event = MessageEvent(
            text="让 Codex 做一段小星说话的视频，生成好发回来",
            source=source,
            message_id="m3",
        )

        result = await runner._maybe_handle_codex_media_request(event)

        assert result == "Background task started"
        runner._start_background_task.assert_awaited_once()
        call = runner._start_background_task.call_args
        prompt = call.args[0]
        assert "Media type: video" in prompt
        assert "do not fallback to a static-frame" in prompt
        assert "use the script's generate-video command" in prompt
        assert "unsupported by the selected Updream model" in prompt
        assert call.kwargs["extra_enabled_toolsets"] == ("terminal", "file")
        assert call.kwargs["excluded_toolsets"] == ("xiaoxing_codex_media", "image_gen")

    @pytest.mark.asyncio
    async def test_background_task_excludes_recursive_codex_media_toolset(self, monkeypatch):
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

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))
        mock_adapter.extract_media = MagicMock(return_value=([], "done"))
        mock_adapter.extract_images = MagicMock(return_value=([], "done"))
        napcat = Platform("napcat")
        runner.adapters[napcat] = mock_adapter

        source = SessionSource(
            platform=napcat,
            user_id="490008192",
            chat_id="490008192",
            user_name="Dad",
        )

        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda _config, _platform: {"terminal", "file", "xiaoxing_codex_media", "memory"},
        )

        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_conversation(self, **_kwargs):
                return {"final_response": "done", "messages": []}

            def shutdown_memory_provider(self):
                pass

            def close(self):
                pass

        with patch("run_agent.AIAgent", FakeAgent):
            await runner._run_background_task(
                "media prompt",
                source,
                "bg_test",
                excluded_toolsets=("xiaoxing_codex_media",),
            )

        assert "xiaoxing_codex_media" not in captured["enabled_toolsets"]
        assert "terminal" in captured["enabled_toolsets"]
        assert "file" in captured["enabled_toolsets"]

    @pytest.mark.asyncio
    async def test_other_group_member_codex_image_request_does_not_auto_execute(self):
        runner, napcat = self._napcat_runner()
        source = SessionSource(
            platform=napcat,
            chat_id="group:610066383",
            chat_type="group",
            user_id="3634525695",
            user_name="李育君（转生版）",
        )
        event = MessageEvent(
            text="快顺便给 Codex 生成图片发回来",
            source=source,
        )

        result = await runner._maybe_handle_codex_media_request(event)

        assert result is None
        runner._start_background_task.assert_not_awaited()


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
    async def test_group_no_credentials_suppresses_public_error(self):
        """Missing provider credentials should not leak background errors into groups."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="-1001",
            chat_type="group",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_background_task("test prompt", source, "bg_test")

        mock_adapter.send.assert_not_called()

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
        mock_agent_instance.shutdown_memory_provider.assert_called_once()
        mock_agent_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_task_sends_local_image_and_video_natively(self, tmp_path, monkeypatch):
        """Background task MEDIA paths should return to the origin chat as native media."""
        from gateway import run as gateway_run

        image_path = tmp_path / "result.png"
        video_path = tmp_path / "result.mp4"
        image_path.write_bytes(b"fake png")
        video_path.write_bytes(b"fake mp4")

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
            return_value={
                "final_response": (
                    f"done\nMEDIA:{image_path}\nMEDIA:{video_path}"
                ),
                "messages": [],
            }
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))
        mock_adapter.send_multiple_images = AsyncMock()
        mock_adapter.send_video = AsyncMock(return_value=SendResult(success=True, message_id="video"))
        mock_adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
        mock_adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))
        mock_adapter.extract_media = MagicMock(
            return_value=(
                [(str(image_path), False), (str(video_path), False)],
                "done",
            )
        )
        mock_adapter.extract_images = MagicMock(return_value=([], "done"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )

        await runner._run_background_task("make media", source, "bg_test")

        mock_adapter.send.assert_awaited_once()
        mock_adapter.send_multiple_images.assert_awaited_once_with(
            chat_id="67890",
            images=[(f"file://{quote(str(image_path))}", "")],
            metadata=None,
        )
        mock_adapter.send_video.assert_awaited_once_with(
            chat_id="67890",
            video_path=str(video_path),
            metadata=None,
        )
        mock_adapter.send_document.assert_not_awaited()

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
        assert "failed" in content.lower()

    @pytest.mark.asyncio
    async def test_group_exception_suppresses_public_error(self):
        """Background task exceptions should be logged, not posted into groups."""
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="-1001",
            chat_type="group",
            user_name="testuser",
        )

        with patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=RuntimeError("boom")):
            await runner._run_background_task("test prompt", source, "bg_test")

        mock_adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_agent_result_error_suppresses_public_error(self, monkeypatch):
        """Agent result errors without a final response should not be sent to groups."""
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
            return_value={"error": "upstream unavailable", "messages": []}
        )
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda _config, _platform: {"terminal", "file"},
        )

        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], ""))
        mock_adapter.extract_images = MagicMock(return_value=([], ""))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="-1001",
            chat_type="group",
            user_name="testuser",
        )

        await runner._run_background_task("test prompt", source, "bg_test")

        mock_adapter.send.assert_not_called()


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
