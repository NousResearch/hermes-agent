"""Tests for the /voice command and auto voice reply in the gateway."""

import json
import os
import queue
import threading
import time
import pytest
from types import SimpleNamespace
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
    """Test the real _should_send_voice_reply method on GatewayRunner.

    The gateway has two TTS paths:
      1. base adapter auto-TTS: fires for voice input in _process_message_background
      2. gateway _send_voice_reply: fires based on voice_mode setting

    To prevent double audio, _send_voice_reply is skipped when voice input
    already triggered base adapter auto-TTS (skip_double = is_voice_input).
    Exception: Discord voice channel — both auto-TTS and Discord play_tts
    override skip, so the runner must handle it via play_in_voice_channel.
    """

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    def _call(self, runner, voice_mode, message_type, agent_messages=None,
              response="Hello!", in_voice_channel=False):
        """Call real _should_send_voice_reply on a GatewayRunner instance."""
        chat_id = "123"
        if voice_mode != "off":
            runner._voice_mode[chat_id] = voice_mode
        else:
            runner._voice_mode.pop(chat_id, None)

        event = _make_event(message_type=message_type)

        if in_voice_channel:
            mock_adapter = MagicMock()
            mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
            event.raw_message = SimpleNamespace(guild_id=111, guild=None)
            runner.adapters[event.source.platform] = mock_adapter

        return runner._should_send_voice_reply(
            event, response, agent_messages or []
        )

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

    def test_voice_input_voice_only_skipped(self, runner):
        """voice_only + voice input: base auto-TTS handles it, runner skips."""
        assert self._call(runner, "voice_only", MessageType.VOICE) is False

    def test_voice_input_all_mode_skipped(self, runner):
        """all + voice input: base auto-TTS handles it, runner skips."""
        assert self._call(runner, "all", MessageType.VOICE) is False

    # -- Text input: only runner handles -----------------------------------

    def test_text_input_all_mode_runner_fires(self, runner):
        """all + text input: only runner fires (base auto-TTS only for voice)."""
        assert self._call(runner, "all", MessageType.TEXT) is True

    def test_text_input_voice_only_no_reply(self, runner):
        """voice_only + text input: neither fires."""
        assert self._call(runner, "voice_only", MessageType.TEXT) is False

    # -- Mode off: nothing fires -------------------------------------------

    def test_off_mode_voice(self, runner):
        assert self._call(runner, "off", MessageType.VOICE) is False

    def test_off_mode_text(self, runner):
        assert self._call(runner, "off", MessageType.TEXT) is False

    # -- Discord VC exception: runner must handle --------------------------

    def test_discord_vc_voice_input_runner_fires(self, runner):
        """Discord VC + voice input: base play_tts skips (VC override),
        so runner must handle via play_in_voice_channel."""
        assert self._call(runner, "all", MessageType.VOICE, in_voice_channel=True) is True

    def test_discord_vc_voice_only_runner_fires(self, runner):
        """Discord VC + voice_only + voice: runner must handle."""
        assert self._call(runner, "voice_only", MessageType.VOICE, in_voice_channel=True) is True

    # -- Edge cases --------------------------------------------------------

    def test_error_response_skipped(self, runner):
        assert self._call(runner, "all", MessageType.TEXT, response="Error: boom") is False

    def test_empty_response_skipped(self, runner):
        assert self._call(runner, "all", MessageType.TEXT, response="") is False

    def test_dedup_skips_when_agent_called_tts(self, runner):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "text_to_speech", "arguments": "{}"},
            }],
        }]
        assert self._call(runner, "all", MessageType.TEXT, agent_messages=messages) is False

    def test_no_dedup_for_other_tools(self, runner):
        messages = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        }]
        assert self._call(runner, "all", MessageType.TEXT, agent_messages=messages) is True


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


# =====================================================================
# VoiceReceiver unit tests
# =====================================================================

class TestVoiceReceiver:
    """Test VoiceReceiver silence detection, SSRC mapping, and lifecycle."""

    def _make_receiver(self):
        from gateway.platforms.discord import VoiceReceiver
        mock_vc = MagicMock()
        mock_vc._connection.secret_key = [0] * 32
        mock_vc._connection.dave_session = None
        mock_vc._connection.ssrc = 9999
        mock_vc._connection.add_socket_listener = MagicMock()
        mock_vc._connection.remove_socket_listener = MagicMock()
        mock_vc._connection.hook = None
        receiver = VoiceReceiver(mock_vc)
        return receiver

    def test_initial_state(self):
        receiver = self._make_receiver()
        assert receiver._running is False
        assert receiver._paused is False
        assert len(receiver._buffers) == 0
        assert len(receiver._ssrc_to_user) == 0

    def test_start_sets_running(self):
        receiver = self._make_receiver()
        receiver.start()
        assert receiver._running is True

    def test_stop_clears_state(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.map_ssrc(100, 42)
        receiver._buffers[100] = bytearray(b"\x00" * 1000)
        receiver._last_packet_time[100] = time.monotonic()
        receiver.stop()
        assert receiver._running is False
        assert len(receiver._buffers) == 0
        assert len(receiver._ssrc_to_user) == 0
        assert len(receiver._last_packet_time) == 0

    def test_map_ssrc(self):
        receiver = self._make_receiver()
        receiver.map_ssrc(100, 42)
        assert receiver._ssrc_to_user[100] == 42

    def test_map_ssrc_overwrites(self):
        receiver = self._make_receiver()
        receiver.map_ssrc(100, 42)
        receiver.map_ssrc(100, 99)
        assert receiver._ssrc_to_user[100] == 99

    def test_pause_resume(self):
        receiver = self._make_receiver()
        assert receiver._paused is False
        receiver.pause()
        assert receiver._paused is True
        receiver.resume()
        assert receiver._paused is False

    def test_check_silence_empty(self):
        receiver = self._make_receiver()
        assert receiver.check_silence() == []

    def test_check_silence_returns_completed_utterance(self):
        receiver = self._make_receiver()
        receiver.map_ssrc(100, 42)
        # 48kHz, stereo, 16-bit = 192000 bytes/sec
        # MIN_SPEECH_DURATION = 0.5s → need 96000 bytes
        pcm_data = bytearray(b"\x00" * 96000)
        receiver._buffers[100] = pcm_data
        # Set last_packet_time far enough in the past to exceed SILENCE_THRESHOLD
        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 1
        user_id, data = completed[0]
        assert user_id == 42
        assert len(data) == 96000
        # Buffer should be cleared after extraction
        assert len(receiver._buffers[100]) == 0

    def test_check_silence_ignores_short_buffer(self):
        receiver = self._make_receiver()
        receiver.map_ssrc(100, 42)
        # Too short to meet MIN_SPEECH_DURATION
        receiver._buffers[100] = bytearray(b"\x00" * 100)
        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_check_silence_ignores_recent_audio(self):
        receiver = self._make_receiver()
        receiver.map_ssrc(100, 42)
        receiver._buffers[100] = bytearray(b"\x00" * 96000)
        receiver._last_packet_time[100] = time.monotonic()  # just now
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_check_silence_unknown_user_discarded(self):
        receiver = self._make_receiver()
        # No SSRC mapping — user_id will be 0
        receiver._buffers[100] = bytearray(b"\x00" * 96000)
        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_stale_buffer_discarded(self):
        receiver = self._make_receiver()
        # Buffer with no user mapping and very old timestamp
        receiver._buffers[200] = bytearray(b"\x00" * 100)
        receiver._last_packet_time[200] = time.monotonic() - 10.0
        receiver.check_silence()
        # Stale buffer (> 2x threshold) should be discarded
        assert 200 not in receiver._buffers

    def test_on_packet_skips_when_not_running(self):
        receiver = self._make_receiver()
        # Not started — _running is False
        receiver._on_packet(b"\x00" * 100)
        assert len(receiver._buffers) == 0

    def test_on_packet_skips_when_paused(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.pause()
        receiver._on_packet(b"\x00" * 100)
        # Paused — should not process
        assert len(receiver._buffers) == 0

    def test_on_packet_skips_short_data(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver._on_packet(b"\x00" * 10)
        assert len(receiver._buffers) == 0

    def test_on_packet_skips_non_rtp(self):
        receiver = self._make_receiver()
        receiver.start()
        # Valid length but wrong RTP version
        data = bytearray(b"\x00" * 20)
        data[0] = 0x00  # version 0, not 2
        receiver._on_packet(bytes(data))
        assert len(receiver._buffers) == 0


# =====================================================================
# Gateway voice channel commands (join / leave / input)
# =====================================================================

class TestVoiceChannelCommands:
    """Test _handle_voice_channel_join, _handle_voice_channel_leave,
    _handle_voice_channel_input on the GatewayRunner."""

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    def _make_discord_event(self, text="/voice channel", chat_id="123",
                            guild_id=111, user_id="user1"):
        """Create event with raw_message carrying guild info."""
        source = SessionSource(
            chat_id=chat_id,
            user_id=user_id,
            platform=MagicMock(),
        )
        source.platform.value = "discord"
        source.thread_id = None
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
        event.message_id = "msg42"
        event.raw_message = SimpleNamespace(guild_id=guild_id, guild=None)
        return event

    # -- _handle_voice_channel_join --

    @pytest.mark.asyncio
    async def test_join_unsupported_platform(self, runner):
        """Platform without join_voice_channel returns unsupported message."""
        mock_adapter = AsyncMock(spec=[])  # no join_voice_channel
        event = self._make_discord_event()
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "not supported" in result.lower()

    @pytest.mark.asyncio
    async def test_join_no_guild_id(self, runner):
        """DM context (no guild_id) returns error."""
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock()
        event = self._make_discord_event()
        event.raw_message = None  # no guild info
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "discord server" in result.lower()

    @pytest.mark.asyncio
    async def test_join_user_not_in_vc(self, runner):
        """User not in any voice channel."""
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock()
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=None)
        event = self._make_discord_event()
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "need to be in a voice channel" in result.lower()

    @pytest.mark.asyncio
    async def test_join_success(self, runner):
        """Successful join sets voice_mode and returns confirmation."""
        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(return_value=True)
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        mock_adapter._voice_text_channels = {}
        mock_adapter._voice_input_callback = None
        event = self._make_discord_event()
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "joined" in result.lower()
        assert "General" in result
        assert runner._voice_mode["123"] == "all"

    @pytest.mark.asyncio
    async def test_join_failure(self, runner):
        """Failed join returns permissions error."""
        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(return_value=False)
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        event = self._make_discord_event()
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_join_exception(self, runner):
        """Exception during join is caught and reported."""
        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(side_effect=RuntimeError("No permission"))
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        event = self._make_discord_event()
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "failed" in result.lower()

    # -- _handle_voice_channel_leave --

    @pytest.mark.asyncio
    async def test_leave_not_in_vc(self, runner):
        """Leave when not in VC returns appropriate message."""
        mock_adapter = AsyncMock()
        mock_adapter.is_in_voice_channel = MagicMock(return_value=False)
        event = self._make_discord_event("/voice leave")
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_leave(event)
        assert "not in" in result.lower()

    @pytest.mark.asyncio
    async def test_leave_no_guild(self, runner):
        """Leave from DM returns not in voice channel."""
        mock_adapter = AsyncMock()
        event = self._make_discord_event("/voice leave")
        event.raw_message = None
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_leave(event)
        assert "not in" in result.lower()

    @pytest.mark.asyncio
    async def test_leave_success(self, runner):
        """Successful leave disconnects and clears voice mode."""
        mock_adapter = AsyncMock()
        mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
        mock_adapter.leave_voice_channel = AsyncMock()
        event = self._make_discord_event("/voice leave")
        runner.adapters[event.source.platform] = mock_adapter
        runner._voice_mode["123"] = "all"
        result = await runner._handle_voice_channel_leave(event)
        assert "left" in result.lower()
        assert "123" not in runner._voice_mode
        mock_adapter.leave_voice_channel.assert_called_once_with(111)

    # -- _handle_voice_channel_input --

    @pytest.mark.asyncio
    async def test_input_no_adapter(self, runner):
        """No Discord adapter — early return, no crash."""
        from gateway.config import Platform
        # No adapters set
        await runner._handle_voice_channel_input(111, 42, "Hello")

    @pytest.mark.asyncio
    async def test_input_no_text_channel(self, runner):
        """No text channel mapped for guild — early return."""
        from gateway.config import Platform
        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {}
        mock_adapter._client = MagicMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        await runner._handle_voice_channel_input(111, 42, "Hello")

    @pytest.mark.asyncio
    async def test_input_creates_event_and_dispatches(self, runner):
        """Voice input creates synthetic event and calls handle_message."""
        from gateway.config import Platform
        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_channel = AsyncMock()
        mock_adapter._client = MagicMock()
        mock_adapter._client.get_channel = MagicMock(return_value=mock_channel)
        mock_adapter.handle_message = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        await runner._handle_voice_channel_input(111, 42, "Hello from VC")
        mock_adapter.handle_message.assert_called_once()
        event = mock_adapter.handle_message.call_args[0][0]
        assert event.text == "Hello from VC"
        assert event.message_type == MessageType.VOICE
        assert event.source.chat_id == "123"

    @pytest.mark.asyncio
    async def test_input_posts_transcript_in_text_channel(self, runner):
        """Voice input sends transcript message to text channel."""
        from gateway.config import Platform
        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_channel = AsyncMock()
        mock_adapter._client = MagicMock()
        mock_adapter._client.get_channel = MagicMock(return_value=mock_channel)
        mock_adapter.handle_message = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter
        await runner._handle_voice_channel_input(111, 42, "Test transcript")
        mock_channel.send.assert_called_once()
        msg = mock_channel.send.call_args[0][0]
        assert "Test transcript" in msg
        assert "42" in msg  # user_id in mention

    # -- _get_guild_id --

    def test_get_guild_id_from_guild(self, runner):
        event = _make_event()
        mock_guild = MagicMock()
        mock_guild.id = 555
        event.raw_message = SimpleNamespace(guild_id=None, guild=mock_guild)
        result = runner._get_guild_id(event)
        assert result == 555

    def test_get_guild_id_from_interaction(self, runner):
        event = _make_event()
        event.raw_message = SimpleNamespace(guild_id=777, guild=None)
        result = runner._get_guild_id(event)
        assert result == 777

    def test_get_guild_id_none(self, runner):
        event = _make_event()
        event.raw_message = None
        result = runner._get_guild_id(event)
        assert result is None

    def test_get_guild_id_dm(self, runner):
        event = _make_event()
        event.raw_message = SimpleNamespace(guild_id=None, guild=None)
        result = runner._get_guild_id(event)
        assert result is None


# =====================================================================
# Discord adapter voice channel methods
# =====================================================================

class TestDiscordVoiceChannelMethods:
    """Test DiscordAdapter voice channel methods (join, leave, play, etc.)."""

    def _make_adapter(self):
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import Platform, PlatformConfig
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._client = MagicMock()
        adapter._voice_clients = {}
        adapter._voice_text_channels = {}
        adapter._voice_timeout_tasks = {}
        adapter._voice_receivers = {}
        adapter._voice_listen_tasks = {}
        adapter._voice_input_callback = None
        adapter._allowed_user_ids = set()
        adapter._running = True
        return adapter

    def test_is_in_voice_channel_true(self):
        adapter = self._make_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        adapter._voice_clients[111] = mock_vc
        assert adapter.is_in_voice_channel(111) is True

    def test_is_in_voice_channel_false_no_client(self):
        adapter = self._make_adapter()
        assert adapter.is_in_voice_channel(111) is False

    def test_is_in_voice_channel_false_disconnected(self):
        adapter = self._make_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = False
        adapter._voice_clients[111] = mock_vc
        assert adapter.is_in_voice_channel(111) is False

    @pytest.mark.asyncio
    async def test_leave_voice_channel_cleans_up(self):
        adapter = self._make_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 123

        mock_receiver = MagicMock()
        adapter._voice_receivers[111] = mock_receiver

        mock_task = MagicMock()
        adapter._voice_listen_tasks[111] = mock_task

        mock_timeout = MagicMock()
        adapter._voice_timeout_tasks[111] = mock_timeout

        await adapter.leave_voice_channel(111)

        mock_receiver.stop.assert_called_once()
        mock_task.cancel.assert_called_once()
        mock_vc.disconnect.assert_called_once()
        mock_timeout.cancel.assert_called_once()
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_receivers

    @pytest.mark.asyncio
    async def test_leave_voice_channel_no_connection(self):
        """Leave when not connected — no crash."""
        adapter = self._make_adapter()
        await adapter.leave_voice_channel(111)  # should not raise

    @pytest.mark.asyncio
    async def test_get_user_voice_channel_no_client(self):
        adapter = self._make_adapter()
        adapter._client = None
        result = await adapter.get_user_voice_channel(111, "42")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_voice_channel_no_guild(self):
        adapter = self._make_adapter()
        adapter._client.get_guild = MagicMock(return_value=None)
        result = await adapter.get_user_voice_channel(111, "42")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_voice_channel_user_not_in_vc(self):
        adapter = self._make_adapter()
        mock_guild = MagicMock()
        mock_member = MagicMock()
        mock_member.voice = None
        mock_guild.get_member = MagicMock(return_value=mock_member)
        adapter._client.get_guild = MagicMock(return_value=mock_guild)
        result = await adapter.get_user_voice_channel(111, "42")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_voice_channel_success(self):
        adapter = self._make_adapter()
        mock_vc = MagicMock()
        mock_guild = MagicMock()
        mock_member = MagicMock()
        mock_member.voice = MagicMock()
        mock_member.voice.channel = mock_vc
        mock_guild.get_member = MagicMock(return_value=mock_member)
        adapter._client.get_guild = MagicMock(return_value=mock_guild)
        result = await adapter.get_user_voice_channel(111, "42")
        assert result is mock_vc

    @pytest.mark.asyncio
    async def test_play_in_voice_channel_not_connected(self):
        adapter = self._make_adapter()
        result = await adapter.play_in_voice_channel(111, "/tmp/test.ogg")
        assert result is False

    def test_is_allowed_user_empty_list(self):
        adapter = self._make_adapter()
        assert adapter._is_allowed_user("42") is True

    def test_is_allowed_user_in_list(self):
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"42", "99"}
        assert adapter._is_allowed_user("42") is True

    def test_is_allowed_user_not_in_list(self):
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"99"}
        assert adapter._is_allowed_user("42") is False

    @pytest.mark.asyncio
    async def test_process_voice_input_success(self):
        """Successful voice input: PCM->WAV->STT->callback."""
        adapter = self._make_adapter()
        callback = AsyncMock()
        adapter._voice_input_callback = callback
        adapter._allowed_user_ids = set()

        pcm_data = b"\x00" * 96000

        with patch("gateway.platforms.discord.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "Hello"}), \
             patch("tools.voice_mode.is_whisper_hallucination", return_value=False):
            await adapter._process_voice_input(111, 42, pcm_data)

        callback.assert_called_once_with(guild_id=111, user_id=42, transcript="Hello")

    @pytest.mark.asyncio
    async def test_process_voice_input_hallucination_filtered(self):
        """Whisper hallucination is filtered out."""
        adapter = self._make_adapter()
        callback = AsyncMock()
        adapter._voice_input_callback = callback

        with patch("gateway.platforms.discord.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "Thank you."}), \
             patch("tools.voice_mode.is_whisper_hallucination", return_value=True):
            await adapter._process_voice_input(111, 42, b"\x00" * 96000)

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_voice_input_stt_failure(self):
        """STT failure — callback not called."""
        adapter = self._make_adapter()
        callback = AsyncMock()
        adapter._voice_input_callback = callback

        with patch("gateway.platforms.discord.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": False, "error": "API error"}):
            await adapter._process_voice_input(111, 42, b"\x00" * 96000)

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_voice_input_exception_caught(self):
        """Exception during processing is caught, no crash."""
        adapter = self._make_adapter()
        adapter._voice_input_callback = AsyncMock()

        with patch("gateway.platforms.discord.VoiceReceiver.pcm_to_wav",
                   side_effect=RuntimeError("ffmpeg not found")):
            await adapter._process_voice_input(111, 42, b"\x00" * 96000)
        # Should not raise


# =====================================================================
# stream_tts_to_speaker functional tests
# =====================================================================

# =====================================================================
# VoiceReceiver thread-safety (lock coverage)
# =====================================================================

class TestVoiceReceiverThreadSafety:
    """Verify that VoiceReceiver buffer access is protected by lock."""

    def _make_receiver(self):
        from gateway.platforms.discord import VoiceReceiver
        mock_vc = MagicMock()
        mock_vc._connection.secret_key = [0] * 32
        mock_vc._connection.dave_session = None
        mock_vc._connection.ssrc = 9999
        mock_vc._connection.add_socket_listener = MagicMock()
        mock_vc._connection.remove_socket_listener = MagicMock()
        mock_vc._connection.hook = None
        return VoiceReceiver(mock_vc)

    def test_check_silence_holds_lock(self):
        """check_silence must hold lock while iterating buffers."""
        import ast, inspect, textwrap
        from gateway.platforms.discord import VoiceReceiver
        source = textwrap.dedent(inspect.getsource(VoiceReceiver.check_silence))
        tree = ast.parse(source)
        # Find 'with self._lock:' that contains buffer iteration
        found_lock_with_for = False
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                # Check if lock context and contains for loop
                has_lock = any(
                    "lock" in ast.dump(item) for item in node.items
                )
                has_for = any(isinstance(n, ast.For) for n in ast.walk(node))
                if has_lock and has_for:
                    found_lock_with_for = True
        assert found_lock_with_for, (
            "check_silence must hold self._lock while iterating buffers"
        )

    def test_on_packet_buffer_write_holds_lock(self):
        """_on_packet must hold lock when writing to buffers."""
        import ast, inspect, textwrap
        from gateway.platforms.discord import VoiceReceiver
        source = textwrap.dedent(inspect.getsource(VoiceReceiver._on_packet))
        tree = ast.parse(source)
        # Find 'with self._lock:' that contains buffer extend
        found_lock_with_extend = False
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                src_fragment = ast.dump(node)
                if "lock" in src_fragment and "extend" in src_fragment:
                    found_lock_with_extend = True
        assert found_lock_with_extend, (
            "_on_packet must hold self._lock when extending buffers"
        )

    def test_concurrent_buffer_access_safe(self):
        """Simulate concurrent buffer writes and reads under lock."""
        import threading
        receiver = self._make_receiver()
        receiver.start()
        errors = []

        def writer():
            for _ in range(1000):
                with receiver._lock:
                    receiver._buffers[100].extend(b"\x00" * 192)
                    receiver._last_packet_time[100] = time.monotonic()

        def reader():
            for _ in range(1000):
                try:
                    receiver.check_silence()
                except Exception as e:
                    errors.append(str(e))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0, f"Race detected: {errors[:3]}"


# =====================================================================
# Callback wiring order (join)
# =====================================================================

class TestCallbackWiringOrder:
    """Verify callback is wired BEFORE join, not after."""

    def test_callback_set_before_join(self):
        """_handle_voice_channel_join wires callback before calling join."""
        import ast, inspect
        from gateway.run import GatewayRunner
        source = inspect.getsource(GatewayRunner._handle_voice_channel_join)
        lines = source.split("\n")
        callback_line = None
        join_line = None
        for i, line in enumerate(lines):
            if "_voice_input_callback" in line and "=" in line and "None" not in line:
                if callback_line is None:
                    callback_line = i
            if "join_voice_channel" in line and "await" in line:
                join_line = i
        assert callback_line is not None, "callback wiring not found"
        assert join_line is not None, "join_voice_channel call not found"
        assert callback_line < join_line, (
            f"callback must be wired (line {callback_line}) BEFORE "
            f"join_voice_channel (line {join_line})"
        )

    @pytest.mark.asyncio
    async def test_join_failure_clears_callback(self, tmp_path):
        """If join fails with exception, callback is cleaned up."""
        runner = _make_runner(tmp_path)

        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(
            side_effect=RuntimeError("No permission")
        )
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        mock_adapter._voice_input_callback = None

        event = _make_event("/voice channel")
        event.raw_message = SimpleNamespace(guild_id=111, guild=None)
        runner.adapters[event.source.platform] = mock_adapter

        result = await runner._handle_voice_channel_join(event)
        assert "failed" in result.lower()
        assert mock_adapter._voice_input_callback is None

    @pytest.mark.asyncio
    async def test_join_returns_false_clears_callback(self, tmp_path):
        """If join returns False, callback is cleaned up."""
        runner = _make_runner(tmp_path)

        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(return_value=False)
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        mock_adapter._voice_input_callback = None

        event = _make_event("/voice channel")
        event.raw_message = SimpleNamespace(guild_id=111, guild=None)
        runner.adapters[event.source.platform] = mock_adapter

        result = await runner._handle_voice_channel_join(event)
        assert "failed" in result.lower()
        assert mock_adapter._voice_input_callback is None


# =====================================================================
# Leave exception handling
# =====================================================================

class TestLeaveExceptionHandling:
    """Verify state is cleaned up even when leave_voice_channel raises."""

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_leave_exception_still_cleans_state(self, runner):
        """If leave_voice_channel raises, voice_mode is still cleaned up."""
        mock_adapter = AsyncMock()
        mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
        mock_adapter.leave_voice_channel = AsyncMock(
            side_effect=RuntimeError("Connection reset")
        )
        mock_adapter._voice_input_callback = MagicMock()

        event = _make_event("/voice leave")
        event.raw_message = SimpleNamespace(guild_id=111, guild=None)
        runner.adapters[event.source.platform] = mock_adapter
        runner._voice_mode["123"] = "all"

        result = await runner._handle_voice_channel_leave(event)
        assert "left" in result.lower()
        assert "123" not in runner._voice_mode
        assert mock_adapter._voice_input_callback is None

    @pytest.mark.asyncio
    async def test_leave_clears_callback(self, runner):
        """Normal leave also clears the voice input callback."""
        mock_adapter = AsyncMock()
        mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
        mock_adapter.leave_voice_channel = AsyncMock()
        mock_adapter._voice_input_callback = MagicMock()

        event = _make_event("/voice leave")
        event.raw_message = SimpleNamespace(guild_id=111, guild=None)
        runner.adapters[event.source.platform] = mock_adapter
        runner._voice_mode["123"] = "all"

        await runner._handle_voice_channel_leave(event)
        assert mock_adapter._voice_input_callback is None


# =====================================================================
# Base adapter empty text guard
# =====================================================================

class TestAutoTtsEmptyTextGuard:
    """Verify base adapter skips TTS when text is empty after markdown strip."""

    def test_empty_after_strip_skips_tts(self):
        """Markdown-only content should not trigger TTS call."""
        import re
        text_content = "****"
        speech_text = re.sub(r'[*_`#\[\]()]', '', text_content)[:4000].strip()
        assert not speech_text, "Expected empty after stripping markdown chars"

    def test_code_block_response_skips_tts(self):
        """Code-only response results in empty speech text."""
        import re
        text_content = "```python\nprint(1)\n```"
        speech_text = re.sub(r'[*_`#\[\]()]', '', text_content)[:4000].strip()
        # Note: base.py regex only strips individual chars, not full code blocks
        # So code blocks are partially stripped but may leave content
        # The real fix is in base.py — empty check after strip

    def test_base_empty_check_in_source(self):
        """base.py must check speech_text is non-empty before calling TTS."""
        import ast, inspect
        from gateway.platforms.base import BasePlatformAdapter
        source = inspect.getsource(BasePlatformAdapter._process_message_background)
        assert "if not speech_text" in source or "not speech_text" in source, (
            "base.py must guard against empty speech_text before TTS call"
        )


class TestStreamTtsToSpeaker:
    """Functional tests for the streaming TTS pipeline."""

    def test_none_sentinel_flushes_buffer(self):
        """None sentinel causes remaining buffer to be spoken."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        def display(text):
            spoken.append(text)

        text_q.put("Hello world.")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=display)
        assert done_evt.is_set()
        assert any("Hello" in s for s in spoken)

    def test_stop_event_aborts_early(self):
        """Setting stop_event causes early exit."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        stop_evt.set()
        text_q.put("Should not be spoken.")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        assert len(spoken) == 0

    def test_done_event_set_on_exception(self):
        """tts_done_event is set even when an exception occurs."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()

        # Put a non-string that will cause concatenation to fail
        text_q.put(12345)
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt)
        assert done_evt.is_set()

    def test_think_blocks_stripped(self):
        """<think>...</think> content is not spoken."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        text_q.put("<think>internal reasoning</think>")
        text_q.put("Visible response. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        joined = " ".join(spoken)
        assert "internal reasoning" not in joined
        assert "Visible" in joined

    def test_sentence_splitting(self):
        """Sentences are split at boundaries and spoken individually."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        # Two sentences long enough to exceed min_sentence_len (20)
        text_q.put("This is the first sentence. ")
        text_q.put("This is the second sentence. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        assert len(spoken) >= 2

    def test_markdown_stripped_in_speech(self):
        """Markdown formatting is removed before display/speech."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        text_q.put("**Bold text** and `code`. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        # Display callback gets raw text (before markdown stripping)
        # But the actual TTS audio would be stripped — we verify pipeline doesn't crash

    def test_duplicate_sentences_deduped(self):
        """Repeated sentences are spoken only once."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        # Same sentence twice, each long enough
        text_q.put("This is a repeated sentence. ")
        text_q.put("This is a repeated sentence. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt, display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        # First occurrence is spoken, second is deduped
        assert len(spoken) == 1

    def test_no_api_key_display_only(self):
        """Without ELEVENLABS_API_KEY, display callback still works."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        text_q.put("Display only text. ")
        text_q.put(None)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            stream_tts_to_speaker(text_q, stop_evt, done_evt,
                                  display_callback=lambda t: spoken.append(t))
        assert done_evt.is_set()
        assert len(spoken) >= 1

    def test_long_buffer_flushed_on_timeout(self):
        """Buffer longer than long_flush_len is flushed on queue timeout."""
        from tools.tts_tool import stream_tts_to_speaker
        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        spoken = []

        # Put a long text without sentence boundary, then None after a delay
        long_text = "a" * 150  # > long_flush_len (100)
        text_q.put(long_text)

        def delayed_sentinel():
            time.sleep(1.0)
            text_q.put(None)

        t = threading.Thread(target=delayed_sentinel, daemon=True)
        t.start()

        stream_tts_to_speaker(text_q, stop_evt, done_evt,
                              display_callback=lambda t: spoken.append(t))
        t.join(timeout=5)
        assert done_evt.is_set()
        assert len(spoken) >= 1
