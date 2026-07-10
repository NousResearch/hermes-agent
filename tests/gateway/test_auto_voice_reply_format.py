"""Tests for gateway auto-TTS voice reply audio format selection."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class TestAutoVoiceReplyFormat:
    @pytest.mark.asyncio
    async def test_telegram_auto_voice_reply_requests_ogg_for_native_voice_bubble(self):
        """Telegram auto-TTS should target Telegram and send returned OGG path."""
        runner = _make_runner()
        adapter = _make_adapter(Platform.TELEGRAM)
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _make_event(Platform.TELEGRAM)
        requested_paths = []

        def fake_tts(*, text, output_path, target_platform, prefer_voice):
            requested_paths.append(output_path)
            assert output_path.endswith(".mp3")
            assert target_platform == Platform.TELEGRAM
            assert prefer_voice is True
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake intermediate mp3")
            actual_path = Path(output_path).with_suffix(".ogg")
            actual_path.write_bytes(b"fake ogg opus")
            return json.dumps({
                "success": True,
                "file_path": str(actual_path),
                "provider": "gemini",
                "voice_compatible": True,
            })

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(event, "hello from auto tts")

        assert requested_paths
        assert requested_paths[0].endswith(".mp3")
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["audio_path"].endswith(".ogg")

    @pytest.mark.asyncio
    async def test_non_telegram_auto_voice_reply_keeps_mp3_default(self):
        """Non-Telegram platforms should keep the current MP3 default."""
        runner = _make_runner()
        adapter = _make_adapter(Platform.SLACK)
        runner.adapters[Platform.SLACK] = adapter
        event = _make_event(Platform.SLACK)
        requested_paths = []

        def fake_tts(*, text, output_path, target_platform, prefer_voice):
            requested_paths.append(output_path)
            assert output_path.endswith(".mp3")
            assert target_platform == Platform.SLACK
            assert prefer_voice is True
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake mp3")
            return json.dumps({
                "success": True,
                "file_path": output_path,
                "provider": "gemini",
                "voice_compatible": False,
            })

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(event, "hello from auto tts")

        assert requested_paths
        assert requested_paths[0].endswith(".mp3")
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["audio_path"].endswith(".mp3")

    @pytest.mark.asyncio
    async def test_telegram_auto_voice_reply_cleans_attempted_path_on_tts_failure(self, tmp_path, monkeypatch):
        """Failed native-provider auto-TTS must clean the rewritten attempted path."""
        monkeypatch.setattr("gateway.platforms.base.tempfile.gettempdir", lambda: str(tmp_path))
        runner = _make_runner()
        adapter = _make_adapter(Platform.TELEGRAM)
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _make_event(Platform.TELEGRAM)
        attempted_paths = []

        def fake_tts(*, text, output_path, target_platform, prefer_voice):
            requested_path = Path(output_path)
            attempted_path = requested_path.with_suffix(".ogg")
            requested_path.write_bytes(b"requested mp3")
            attempted_path.write_bytes(b"partial ogg")
            attempted_paths.append((requested_path, attempted_path))
            return json.dumps({
                "success": False,
                "error": "provider failed",
                "attempted_file_path": str(attempted_path),
            })

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(event, "hello from auto tts")

        requested_path, attempted_path = attempted_paths[0]
        adapter.send_voice.assert_not_awaited()
        assert not requested_path.exists()
        assert not attempted_path.exists()


class TestBaseAdapterAutoVoiceReplyFormat:
    @pytest.mark.asyncio
    async def test_telegram_voice_input_auto_tts_targets_platform_and_sends_returned_path(self):
        """Base adapter voice-input auto-TTS should send provider-returned OGG."""
        adapter = _AutoTtsAdapter(Platform.TELEGRAM)
        adapter._keep_typing = _hold_typing
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="voice-1"))
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="short reply"))
        event = _make_voice_event(Platform.TELEGRAM)
        requested_paths = []
        actual_paths = []

        def fake_tts(*, text, output_path, target_platform, prefer_voice):
            requested_path = Path(output_path)
            requested_paths.append(requested_path)
            assert requested_path.suffix == ".mp3"
            assert target_platform == Platform.TELEGRAM
            assert prefer_voice is True
            assert requested_path.parent.is_dir()
            requested_path.write_bytes(b"intermediate mp3")

            actual_path = requested_path.with_name(f"{requested_path.stem}_actual.ogg")
            actual_paths.append(actual_path)
            actual_path.write_bytes(b"actual ogg")
            return json.dumps({
                "success": True,
                "file_path": str(actual_path),
                "provider": "gemini",
                "voice_compatible": True,
            })

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            side_effect=fake_tts,
        ):
            await adapter._process_message_background(event, build_session_key(event.source))

        assert len(requested_paths) == 1
        assert requested_paths[0].suffix == ".mp3"
        assert actual_paths[0].suffix == ".ogg"
        adapter.play_tts.assert_awaited_once()
        assert adapter.play_tts.await_args.kwargs["audio_path"] == str(actual_paths[0])
        assert not requested_paths[0].exists()
        assert not actual_paths[0].exists()

    @pytest.mark.asyncio
    async def test_non_telegram_voice_input_auto_tts_keeps_mp3_default(self):
        """Base adapter voice-input auto-TTS should preserve MP3 for other platforms."""
        adapter = _AutoTtsAdapter(Platform.SLACK)
        adapter._keep_typing = _hold_typing
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="voice-1"))
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="short reply"))
        event = _make_voice_event(Platform.SLACK)
        requested_paths = []

        def fake_tts(*, text, output_path, target_platform, prefer_voice):
            requested_path = Path(output_path)
            requested_paths.append(requested_path)
            assert requested_path.suffix == ".mp3"
            assert target_platform == Platform.SLACK
            assert prefer_voice is True
            assert requested_path.parent.is_dir()
            requested_path.write_bytes(b"actual mp3")
            return json.dumps({
                "success": True,
                "file_path": str(requested_path),
                "provider": "gemini",
                "voice_compatible": False,
            })

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            side_effect=fake_tts,
        ):
            await adapter._process_message_background(event, build_session_key(event.source))

        assert len(requested_paths) == 1
        assert requested_paths[0].suffix == ".mp3"
        adapter.play_tts.assert_awaited_once()
        assert adapter.play_tts.await_args.kwargs["audio_path"] == str(requested_paths[0])
        assert not requested_paths[0].exists()

    @pytest.mark.asyncio
    async def test_telegram_voice_input_auto_tts_cleans_attempted_path_on_tts_failure(self, tmp_path, monkeypatch):
        """Base adapter auto-TTS should clean attempted paths from failed TTS results."""
        monkeypatch.setattr("gateway.platforms.base.tempfile.gettempdir", lambda: str(tmp_path))
        adapter = _AutoTtsAdapter(Platform.TELEGRAM)
        adapter._keep_typing = _hold_typing
        adapter._should_auto_tts_for_chat = lambda _chat_id: True
        adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="voice-1"))
        adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="short reply"))
        event = _make_voice_event(Platform.TELEGRAM)
        attempted_paths = []

        def fake_tts(*, text, output_path, target_platform, prefer_voice):
            requested_path = Path(output_path)
            attempted_path = requested_path.with_suffix(".ogg")
            requested_path.write_bytes(b"requested mp3")
            attempted_path.write_bytes(b"partial ogg")
            attempted_paths.append((requested_path, attempted_path))
            return json.dumps({
                "success": False,
                "error": "provider failed",
                "attempted_file_path": str(attempted_path),
            })

        with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
            "tools.tts_tool.text_to_speech_tool",
            side_effect=fake_tts,
        ):
            await adapter._process_message_background(event, build_session_key(event.source))

        requested_path, attempted_path = attempted_paths[0]
        adapter.play_tts.assert_not_awaited()
        assert not requested_path.exists()
        assert not attempted_path.exists()


def _make_runner() -> GatewayRunner:
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
    return runner


def _make_adapter(platform: Platform) -> MagicMock:
    adapter = MagicMock()
    adapter.platform = platform
    adapter.send_voice = AsyncMock()
    return adapter


def _make_event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=platform,
            chat_id="123",
            user_id="u1",
            user_name="User",
        ),
        message_id="456",
    )


class _AutoTtsAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="text-1")

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


async def _hold_typing(_chat_id, interval=2.0, metadata=None):
    await asyncio.Event().wait()


def _make_voice_event(platform: Platform) -> MessageEvent:
    return MessageEvent(
        text="voice input",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=platform,
            chat_id="123",
            user_id="u1",
            user_name="User",
        ),
        message_id="voice-456",
    )
