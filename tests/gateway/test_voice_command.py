"""Tests for the /voice command and auto voice reply in the gateway."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text: str = "", message_type=MessageType.TEXT, chat_id="123") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=MagicMock(),
    )
    source.platform.value = "telegram"
    source.thread_id = None
    event = MessageEvent(text=text, message_type=message_type, source=source)
    event.message_id = "msg42"
    return event


def _make_runner(tmp_path):
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    return runner


# =====================================================================
# /voice command handler
# =====================================================================

class TestHandleVoiceCommand:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_voice_on(self, runner):
        event = _make_event("/voice on")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_voice_off(self, runner):
        runner._voice_mode["123"] = "voice_only"
        event = _make_event("/voice off")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert "123" not in runner._voice_mode

    @pytest.mark.asyncio
    async def test_voice_tts(self, runner):
        event = _make_event("/voice tts")
        result = await runner._handle_voice_command(event)
        assert "tts" in result.lower()
        assert runner._voice_mode["123"] == "all"

    @pytest.mark.asyncio
    async def test_voice_status_off(self, runner):
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "off" in result.lower()

    @pytest.mark.asyncio
    async def test_voice_status_on(self, runner):
        runner._voice_mode["123"] = "voice_only"
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "voice reply" in result.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_to_on(self, runner):
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_toggle_on_to_off(self, runner):
        runner._voice_mode["123"] = "voice_only"
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert "123" not in runner._voice_mode

    @pytest.mark.asyncio
    async def test_persistence_saved(self, runner):
        event = _make_event("/voice on")
        await runner._handle_voice_command(event)
        assert runner._VOICE_MODE_PATH.exists()
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_persistence_loaded(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({"456": "all"}))
        loaded = runner._load_voice_modes()
        assert loaded == {"456": "all"}

    @pytest.mark.asyncio
    async def test_per_chat_isolation(self, runner):
        e1 = _make_event("/voice on", chat_id="aaa")
        e2 = _make_event("/voice tts", chat_id="bbb")
        await runner._handle_voice_command(e1)
        await runner._handle_voice_command(e2)
        assert runner._voice_mode["aaa"] == "voice_only"
        assert runner._voice_mode["bbb"] == "all"


# =====================================================================
# Auto voice reply decision logic
# =====================================================================

class TestAutoVoiceReply:
    """Test the should_voice_reply decision logic (extracted from _handle_message).

    The gateway has two TTS paths:
      1. base adapter auto-TTS: fires for voice input in _process_message_background
      2. gateway _send_voice_reply: fires based on voice_mode setting

    To prevent double audio, _send_voice_reply is skipped when voice input
    already triggered base adapter auto-TTS (skip_double = is_voice_input).
    Exception: Discord voice channel — both auto-TTS and Discord play_tts
    override skip, so the runner must handle it via play_in_voice_channel.
    """

    def _should_reply(self, voice_mode, message_type, agent_messages=None,
                      response="Hello!", in_voice_channel=False):
        """Replicate the auto voice reply decision from _handle_message."""
        if not response or response.startswith("Error:"):
            return False

        is_voice_input = (message_type == MessageType.VOICE)
        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
        )
        if not should:
            return False

        # Dedup: agent already called TTS tool
        if agent_messages:
            has_agent_tts = any(
                msg.get("role") == "assistant"
                and any(
                    tc.get("function", {}).get("name") == "text_to_speech"
                    for tc in (msg.get("tool_calls") or [])
                )
                for msg in agent_messages
            )
            if has_agent_tts:
                return False

        # Dedup: base adapter auto-TTS already handles voice input.
        # Exception: in voice channel, Discord play_tts also skips,
        # so the runner must handle VC playback.
        skip_double = is_voice_input
        if skip_double and in_voice_channel:
            skip_double = False
        if skip_double:
            return False

        return True

    # -- Full platform x input x mode matrix --------------------------------
    #
    # Legend:
    #   base = base adapter auto-TTS (play_tts)
    #   runner = gateway _send_voice_reply
    #
    # | Platform      | Input | Mode       | base | runner | Expected     |
    # |---------------|-------|------------|------|--------|--------------|
    # | Telegram      | voice | off        | yes  | skip   | 1 audio      |
    # | Telegram      | voice | voice_only | yes  | skip*  | 1 audio      |
    # | Telegram      | voice | all        | yes  | skip*  | 1 audio      |
    # | Telegram      | text  | off        | skip | skip   | 0 audio      |
    # | Telegram      | text  | voice_only | skip | skip   | 0 audio      |
    # | Telegram      | text  | all        | skip | yes    | 1 audio      |
    # | Discord text  | voice | all        | yes  | skip*  | 1 audio      |
    # | Discord text  | text  | all        | skip | yes    | 1 audio      |
    # | Discord VC    | voice | all        | skip†| yes    | 1 audio (VC) |
    # | Web UI        | voice | off        | yes  | skip   | 1 audio      |
    # | Web UI        | voice | all        | yes  | skip*  | 1 audio      |
    # | Web UI        | text  | all        | skip | yes    | 1 audio      |
    # | Slack         | voice | all        | yes  | skip*  | 1 audio      |
    # | Slack         | text  | all        | skip | yes    | 1 audio      |
    #
    # * skip_double: voice input → base already handles
    # † Discord play_tts override skips when in VC

    # -- Telegram/Slack/Web: voice input, base handles ---------------------

    def test_voice_input_voice_only_skipped(self):
        """voice_only + voice input: base auto-TTS handles it, runner skips."""
        assert self._should_reply("voice_only", MessageType.VOICE) is False

    def test_voice_input_all_mode_skipped(self):
        """all + voice input: base auto-TTS handles it, runner skips."""
        assert self._should_reply("all", MessageType.VOICE) is False

    # -- Text input: only runner handles -----------------------------------

    def test_text_input_all_mode_runner_fires(self):
        """all + text input: only runner fires (base auto-TTS only for voice)."""
        assert self._should_reply("all", MessageType.TEXT) is True

    def test_text_input_voice_only_no_reply(self):
        """voice_only + text input: neither fires."""
        assert self._should_reply("voice_only", MessageType.TEXT) is False

    # -- Mode off: nothing fires -------------------------------------------

    def test_off_mode_voice(self):
        assert self._should_reply("off", MessageType.VOICE) is False

    def test_off_mode_text(self):
        assert self._should_reply("off", MessageType.TEXT) is False

    # -- Discord VC exception: runner must handle --------------------------

    def test_discord_vc_voice_input_runner_fires(self):
        """Discord VC + voice input: base play_tts skips (VC override),
        so runner must handle via play_in_voice_channel."""
        assert self._should_reply("all", MessageType.VOICE, in_voice_channel=True) is True

    def test_discord_vc_voice_only_runner_fires(self):
        """Discord VC + voice_only + voice: runner must handle."""
        assert self._should_reply("voice_only", MessageType.VOICE, in_voice_channel=True) is True

    # -- Edge cases --------------------------------------------------------

    def test_error_response_skipped(self):
        assert self._should_reply("all", MessageType.TEXT, response="Error: boom") is False

    def test_empty_response_skipped(self):
        assert self._should_reply("all", MessageType.TEXT, response="") is False

    def test_dedup_skips_when_agent_called_tts(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "text_to_speech", "arguments": "{}"},
            }],
        }]
        assert self._should_reply("all", MessageType.TEXT, agent_messages=messages) is False

    def test_no_dedup_for_other_tools(self):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        }]
        assert self._should_reply("all", MessageType.TEXT, agent_messages=messages) is True


# =====================================================================
# _send_voice_reply
# =====================================================================

class TestSendVoiceReply:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_calls_tts_and_send_voice(self, runner):
        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        event = _make_event()
        runner.adapters[event.source.platform] = mock_adapter

        tts_result = json.dumps({"success": True, "file_path": "/tmp/test.ogg"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        mock_adapter.send_voice.assert_called_once()
        call_args = mock_adapter.send_voice.call_args
        assert call_args.kwargs.get("chat_id") == "123"

    @pytest.mark.asyncio
    async def test_empty_text_after_strip_skips(self, runner):
        event = _make_event()

        with patch("tools.tts_tool.text_to_speech_tool") as mock_tts, \
             patch("tools.tts_tool._strip_markdown_for_tts", return_value=""):
            await runner._send_voice_reply(event, "```code only```")

        mock_tts.assert_not_called()

    @pytest.mark.asyncio
    async def test_tts_failure_no_crash(self, runner):
        event = _make_event()
        mock_adapter = AsyncMock()
        runner.adapters[event.source.platform] = mock_adapter
        tts_result = json.dumps({"success": False, "error": "API error"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=False), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello")

        mock_adapter.send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_caught(self, runner):
        event = _make_event()
        with patch("tools.tts_tool.text_to_speech_tool", side_effect=RuntimeError("boom")), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.makedirs"):
            # Should not raise
            await runner._send_voice_reply(event, "Hello")


# =====================================================================
# Discord play_tts skip when in voice channel
# =====================================================================

class TestDiscordPlayTtsSkip:
    """Discord adapter skips play_tts when bot is in a voice channel."""

    def _make_discord_adapter(self):
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import Platform, PlatformConfig
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_text_channels = {}
        adapter._voice_timeout_tasks = {}
        adapter._voice_receivers = {}
        adapter._voice_listen_tasks = {}
        adapter._client = None
        adapter._broadcast = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_play_tts_skipped_when_in_vc(self):
        adapter = self._make_discord_adapter()
        # Simulate bot in voice channel for guild 111, text channel 123
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 123

        result = await adapter.play_tts(chat_id="123", audio_path="/tmp/test.ogg")
        assert result.success is True
        # send_voice should NOT have been called (no client, would fail)

    @pytest.mark.asyncio
    async def test_play_tts_not_skipped_when_not_in_vc(self):
        adapter = self._make_discord_adapter()
        # No voice connection — play_tts falls through to send_voice
        result = await adapter.play_tts(chat_id="123", audio_path="/tmp/test.ogg")
        # send_voice will fail (no client), but play_tts should NOT return early
        assert result.success is False

    @pytest.mark.asyncio
    async def test_play_tts_not_skipped_for_different_channel(self):
        adapter = self._make_discord_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999  # different channel

        result = await adapter.play_tts(chat_id="123", audio_path="/tmp/test.ogg")
        # Different channel — should NOT skip, falls through to send_voice (fails)
        assert result.success is False


# =====================================================================
# Web play_tts sends play_audio (not voice bubble)
# =====================================================================

class TestWebPlayTts:
    """Web adapter play_tts sends invisible play_audio, not a voice bubble."""

    @pytest.mark.asyncio
    async def test_play_tts_sends_play_audio(self, tmp_path):
        from gateway.platforms.web import WebAdapter
        from gateway.config import PlatformConfig

        config = PlatformConfig(enabled=True, extra={
            "port": 0, "host": "127.0.0.1", "token": "tok",
        })
        adapter = WebAdapter(config)
        adapter._broadcast = AsyncMock()
        adapter._media_dir = tmp_path / "media"
        adapter._media_dir.mkdir()

        audio_file = tmp_path / "test.ogg"
        audio_file.write_bytes(b"fake audio")

        result = await adapter.play_tts(chat_id="web", audio_path=str(audio_file))
        assert result.success is True

        payload = adapter._broadcast.call_args[0][0]
        assert payload["type"] == "play_audio"
        assert "/media/" in payload["url"]


# =====================================================================
# Help text + known commands
# =====================================================================

class TestVoiceInHelp:

    def test_voice_in_help_output(self):
        from gateway.run import GatewayRunner
        import inspect
        source = inspect.getsource(GatewayRunner._handle_help_command)
        assert "/voice" in source

    def test_voice_is_known_command(self):
        from gateway.run import GatewayRunner
        import inspect
        source = inspect.getsource(GatewayRunner._handle_message)
        assert '"voice"' in source
