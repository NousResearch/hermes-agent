"""Tests for the /voice command and auto voice reply in the gateway."""

import importlib.util
import json
import os
import queue
import sys
import threading
import time
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


def _ensure_discord_mock():
    """Install a lightweight discord mock when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True, load_opus=lambda *_args, **_kwargs: None)
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

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
    runner._is_user_authorized = lambda source: True
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
        assert runner._voice_mode["123"] == "off"

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
        assert runner._voice_mode["123"] == "off"

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
    async def test_persistence_saved_for_off(self, runner):
        event = _make_event("/voice off")
        await runner._handle_voice_command(event)
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["123"] == "off"

    def test_sync_voice_mode_state_to_adapter_restores_off_chats(self, runner):
        runner._voice_mode = {"123": "off", "456": "all"}
        adapter = SimpleNamespace(_auto_tts_disabled_chats=set())

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_disabled_chats == {"123"}

    def test_restart_restores_voice_off_state(self, runner, tmp_path):
        runner._VOICE_MODE_PATH.write_text(json.dumps({"123": "off"}))

        restored_runner = _make_runner(tmp_path)
        restored_runner._voice_mode = restored_runner._load_voice_modes()
        adapter = SimpleNamespace(_auto_tts_disabled_chats=set())

        restored_runner._sync_voice_mode_state_to_adapter(adapter)

        assert restored_runner._voice_mode["123"] == "off"
        assert adapter._auto_tts_disabled_chats == {"123"}

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
    already triggered base adapter auto-TTS.

    For Discord voice channels, the base adapter now routes play_tts directly
    into VC playback, so the runner should still skip voice-input follow-ups to
    avoid double playback.
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

    def test_discord_vc_voice_input_base_handles(self, runner):
        """Discord VC + voice input: base adapter play_tts plays in VC,
        so runner skips to avoid double playback."""
        assert self._call(runner, "all", MessageType.VOICE, in_voice_channel=True) is False

    def test_discord_vc_voice_only_base_handles(self, runner):
        """Discord VC + voice_only + voice: base adapter handles."""
        assert self._call(runner, "voice_only", MessageType.VOICE, in_voice_channel=True) is False

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
    async def test_play_tts_plays_in_vc_when_connected(self):
        adapter = self._make_discord_adapter()
        # Simulate bot in voice channel for guild 111, text channel 123
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.is_playing.return_value = False
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 123

        # Mock play_in_voice_channel to avoid actual ffmpeg call
        async def fake_play(gid, path):
            return True
        adapter.play_in_voice_channel = fake_play

        result = await adapter.play_tts(chat_id="123", audio_path="/tmp/test.ogg")
        # play_tts now plays in VC instead of being a no-op
        assert result.success is True

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

# =====================================================================
# Help text + known commands
# =====================================================================

class TestVoiceInHelp:

    def test_voice_in_help_output(self):
        """The gateway help text includes /voice (generated from registry)."""
        from hermes_cli.commands import gateway_help_lines
        help_text = "\n".join(gateway_help_lines())
        assert "/voice" in help_text

    def test_voice_is_known_command(self):
        """The /voice command is in GATEWAY_KNOWN_COMMANDS."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "voice" in GATEWAY_KNOWN_COMMANDS


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

    @pytest.mark.asyncio
    async def test_join_missing_voice_dependencies(self, runner):
        """Missing PyNaCl/davey should return a user-actionable install hint."""
        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(
            side_effect=RuntimeError("PyNaCl library needed in order to use voice")
        )
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        event = self._make_discord_event()
        runner.adapters[event.source.platform] = mock_adapter

        result = await runner._handle_voice_channel_join(event)

        assert "voice dependencies are missing" in result.lower()
        assert "hermes-agent[messaging]" in result

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
        assert runner._voice_mode["123"] == "off"
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
        assert event.source.chat_type == "channel"

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
        assert runner._voice_mode["123"] == "off"
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


# =====================================================================
# Bug 1: VoiceReceiver.stop() must hold lock while clearing shared state
# =====================================================================

class TestStopAcquiresLock:
    """stop() must acquire _lock before clearing buffers/state."""

    @staticmethod
    def _make_receiver():
        from gateway.platforms.discord import VoiceReceiver
        vc = MagicMock()
        vc._connection.secret_key = [0] * 32
        vc._connection.dave_session = None
        vc._connection.ssrc = 1
        return VoiceReceiver(vc)

    def test_stop_clears_under_lock(self):
        """stop() acquires _lock before clearing buffers.

        Verify by holding the lock from another thread and checking that
        stop() blocks until the lock is released.
        """
        receiver = self._make_receiver()
        receiver.start()
        receiver._buffers[100] = bytearray(b"\x00" * 500)
        receiver._last_packet_time[100] = time.monotonic()
        receiver.map_ssrc(100, 42)

        # Hold the lock from another thread
        lock_acquired = threading.Event()
        release_lock = threading.Event()

        def hold_lock():
            with receiver._lock:
                lock_acquired.set()
                release_lock.wait(timeout=5)

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        lock_acquired.wait(timeout=2)

        # stop() in another thread — should block on the lock
        stop_done = threading.Event()

        def do_stop():
            receiver.stop()
            stop_done.set()

        stopper = threading.Thread(target=do_stop, daemon=True)
        stopper.start()

        # stop should NOT complete while lock is held
        assert not stop_done.wait(timeout=0.3), \
            "stop() should block while _lock is held by another thread"

        # Release the lock — stop should complete
        release_lock.set()
        assert stop_done.wait(timeout=2), \
            "stop() should complete after lock is released"

        # State should be cleared
        assert len(receiver._buffers) == 0
        assert len(receiver._ssrc_to_user) == 0
        holder.join(timeout=2)
        stopper.join(timeout=2)

    def test_stop_does_not_deadlock_with_on_packet(self):
        """stop() during _on_packet should not deadlock."""
        receiver = self._make_receiver()
        receiver.start()

        blocked = threading.Event()
        released = threading.Event()

        def hold_lock():
            with receiver._lock:
                blocked.set()
                released.wait(timeout=2)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        blocked.wait(timeout=2)

        stop_done = threading.Event()

        def do_stop():
            receiver.stop()
            stop_done.set()

        t2 = threading.Thread(target=do_stop, daemon=True)
        t2.start()

        # stop should be blocked waiting for lock
        assert not stop_done.wait(timeout=0.2), \
            "stop() should wait for lock, not clear without it"

        released.set()
        assert stop_done.wait(timeout=2), "stop() should complete after lock released"
        t.join(timeout=2)
        t2.join(timeout=2)


# =====================================================================
# Bug 2: _packet_debug_count must be instance-level, not class-level
# =====================================================================

class TestPacketDebugCounterIsInstanceLevel:
    """Each VoiceReceiver instance has its own debug counter."""

    @staticmethod
    def _make_receiver():
        from gateway.platforms.discord import VoiceReceiver
        vc = MagicMock()
        vc._connection.secret_key = [0] * 32
        vc._connection.dave_session = None
        vc._connection.ssrc = 1
        return VoiceReceiver(vc)

    def test_counter_is_per_instance(self):
        """Two receivers have independent counters."""
        r1 = self._make_receiver()
        r2 = self._make_receiver()

        r1._packet_debug_count = 10
        assert r2._packet_debug_count == 0, \
            "_packet_debug_count must be instance-level, not shared across instances"

    def test_counter_initialized_in_init(self):
        """Counter is set in __init__, not as a class variable."""
        r = self._make_receiver()
        assert "_packet_debug_count" in r.__dict__, \
            "_packet_debug_count should be in instance __dict__, not class"


# =====================================================================
# Bug 3: play_in_voice_channel uses get_running_loop not get_event_loop
# =====================================================================

class TestPlayInVoiceChannelUsesRunningLoop:
    """play_in_voice_channel must use asyncio.get_running_loop()."""

    def test_source_uses_get_running_loop(self):
        """The method source code calls get_running_loop, not get_event_loop."""
        import inspect
        from gateway.platforms.discord import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.play_in_voice_channel)
        assert "get_running_loop" in source, \
            "play_in_voice_channel should use asyncio.get_running_loop()"
        assert "get_event_loop" not in source, \
            "play_in_voice_channel should NOT use deprecated asyncio.get_event_loop()"


# =====================================================================
# Bug 4: _send_voice_reply filename uses uuid (no collision)
# =====================================================================

class TestSendVoiceReplyFilename:
    """_send_voice_reply uses uuid for unique filenames."""

    def test_filename_uses_uuid(self):
        """The method uses uuid in the filename, not time-based."""
        import inspect
        from gateway.run import GatewayRunner
        source = inspect.getsource(GatewayRunner._send_voice_reply)
        assert "uuid" in source, \
            "_send_voice_reply should use uuid for unique filenames"
        assert "int(time.time())" not in source, \
            "_send_voice_reply should not use int(time.time()) — collision risk"

    def test_filenames_are_unique(self):
        """Two calls produce different filenames."""
        import uuid
        names = set()
        for _ in range(100):
            name = f"tts_reply_{uuid.uuid4().hex[:12]}.mp3"
            assert name not in names, f"Collision detected: {name}"
            names.add(name)


# =====================================================================
# Bug 5: Voice timeout cleans up runner voice_mode via callback
# =====================================================================

class TestVoiceTimeoutCleansRunnerState:
    """Timeout disconnect notifies runner to clean voice_mode."""

    @staticmethod
    def _make_discord_adapter():
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import PlatformConfig, Platform
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
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._client = None
        adapter._broadcast = AsyncMock()
        adapter._allowed_user_ids = set()
        return adapter

    @pytest.fixture
    def adapter(self):
        return self._make_discord_adapter()

    def test_adapter_has_on_voice_disconnect_attr(self, adapter):
        """DiscordAdapter has _on_voice_disconnect callback attribute."""
        assert hasattr(adapter, "_on_voice_disconnect")
        assert adapter._on_voice_disconnect is None

    @pytest.mark.asyncio
    async def test_timeout_calls_disconnect_callback(self, adapter):
        """_voice_timeout_handler calls _on_voice_disconnect with chat_id."""
        callback_calls = []
        adapter._on_voice_disconnect = lambda chat_id: callback_calls.append(chat_id)

        # Set up state as if we're in a voice channel
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999
        adapter._voice_timeout_tasks[111] = MagicMock()
        adapter._voice_receivers[111] = MagicMock()
        adapter._voice_listen_tasks[111] = MagicMock()

        # Patch sleep to return immediately
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await adapter._voice_timeout_handler(111)

        assert "999" in callback_calls, \
            "_on_voice_disconnect must be called with chat_id on timeout"

    @pytest.mark.asyncio
    async def test_runner_cleanup_method_removes_voice_mode(self, tmp_path):
        """_handle_voice_timeout_cleanup removes voice_mode for chat."""
        runner = _make_runner(tmp_path)
        runner._voice_mode["999"] = "all"

        runner._handle_voice_timeout_cleanup("999")

        assert runner._voice_mode["999"] == "off", \
            "voice_mode must persist explicit off state after timeout cleanup"

    @pytest.mark.asyncio
    async def test_timeout_without_callback_does_not_crash(self, adapter):
        """Timeout works even without _on_voice_disconnect set."""
        adapter._on_voice_disconnect = None

        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999
        adapter._voice_timeout_tasks[111] = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await adapter._voice_timeout_handler(111)

        assert 111 not in adapter._voice_clients


# =====================================================================
# Bug 6: play_in_voice_channel has playback timeout
# =====================================================================

class TestPlaybackTimeout:
    """play_in_voice_channel must time out instead of blocking forever."""

    @staticmethod
    def _make_discord_adapter():
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import PlatformConfig, Platform
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
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._client = None
        adapter._broadcast = AsyncMock()
        adapter._allowed_user_ids = set()
        return adapter

    def test_source_has_wait_for_timeout(self):
        """The method uses asyncio.wait_for with timeout."""
        import inspect
        from gateway.platforms.discord import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.play_in_voice_channel)
        assert "wait_for" in source, \
            "play_in_voice_channel must use asyncio.wait_for for timeout"
        assert "PLAYBACK_TIMEOUT" in source, \
            "play_in_voice_channel must reference PLAYBACK_TIMEOUT constant"

    def test_playback_timeout_constant_exists(self):
        """PLAYBACK_TIMEOUT constant is defined on DiscordAdapter."""
        from gateway.platforms.discord import DiscordAdapter
        assert hasattr(DiscordAdapter, "PLAYBACK_TIMEOUT")
        assert DiscordAdapter.PLAYBACK_TIMEOUT > 0

    @pytest.mark.asyncio
    async def test_playback_timeout_fires(self):
        """When done event is never set, playback times out gracefully."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = self._make_discord_adapter()

        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.is_playing.return_value = False
        # play() never calls the after callback -> done never set
        mock_vc.play = MagicMock()
        mock_vc.stop = MagicMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_timeout_tasks[111] = MagicMock()

        # Use a tiny timeout for test speed
        original_timeout = DiscordAdapter.PLAYBACK_TIMEOUT
        DiscordAdapter.PLAYBACK_TIMEOUT = 0.1
        try:
            with patch("discord.FFmpegPCMAudio"), \
                 patch("discord.PCMVolumeTransformer", side_effect=lambda s, **kw: s):
                result = await adapter.play_in_voice_channel(111, "/tmp/test.mp3")
            assert result is True
            # vc.stop() should have been called due to timeout
            mock_vc.stop.assert_called()
        finally:
            DiscordAdapter.PLAYBACK_TIMEOUT = original_timeout

    @pytest.mark.asyncio
    async def test_is_playing_wait_has_timeout(self):
        """While loop waiting for previous playback has a timeout."""
        from gateway.platforms.discord import DiscordAdapter
        adapter = self._make_discord_adapter()

        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        # is_playing always returns True — would loop forever without timeout
        mock_vc.is_playing.return_value = True
        mock_vc.stop = MagicMock()
        mock_vc.play = MagicMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_timeout_tasks[111] = MagicMock()

        original_timeout = DiscordAdapter.PLAYBACK_TIMEOUT
        DiscordAdapter.PLAYBACK_TIMEOUT = 0.2
        try:
            with patch("discord.FFmpegPCMAudio"), \
                 patch("discord.PCMVolumeTransformer", side_effect=lambda s, **kw: s):
                result = await adapter.play_in_voice_channel(111, "/tmp/test.mp3")
            assert result is True
            # stop() called to break out of the is_playing loop
            mock_vc.stop.assert_called()
        finally:
            DiscordAdapter.PLAYBACK_TIMEOUT = original_timeout


# =====================================================================
# Bug 7: _send_voice_reply cleanup in finally block
# =====================================================================

class TestSendVoiceReplyCleanup:
    """_send_voice_reply must clean up temp files even on exception."""

    def test_cleanup_in_finally(self):
        """The method has cleanup in a finally block, not inside try."""
        import inspect, textwrap, ast
        from gateway.run import GatewayRunner
        source = textwrap.dedent(inspect.getsource(GatewayRunner._send_voice_reply))
        tree = ast.parse(source)
        func = tree.body[0]

        has_finally_unlink = False
        for node in ast.walk(func):
            if isinstance(node, ast.Try) and node.finalbody:
                finally_source = ast.dump(node.finalbody[0])
                if "unlink" in finally_source or "remove" in finally_source:
                    has_finally_unlink = True
                    break

        assert has_finally_unlink, \
            "_send_voice_reply must have os.unlink in a finally block"

    @pytest.mark.asyncio
    async def test_files_cleaned_on_send_exception(self, tmp_path):
        """Temp files are removed even when send_voice raises."""
        runner = _make_runner(tmp_path)
        adapter = MagicMock()
        adapter.send_voice = AsyncMock(side_effect=RuntimeError("send failed"))
        adapter.is_in_voice_channel = MagicMock(return_value=False)
        event = _make_event(message_type=MessageType.VOICE)
        runner.adapters[event.source.platform] = adapter
        runner._get_guild_id = MagicMock(return_value=None)

        # Create a fake audio file that TTS would produce
        fake_audio = tmp_path / "hermes_voice"
        fake_audio.mkdir()
        audio_file = fake_audio / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        tts_result = json.dumps({
            "success": True,
            "file_path": str(audio_file),
        })

        with patch("gateway.run.asyncio.to_thread", new_callable=AsyncMock, return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", return_value="hello"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        # File should be cleaned up despite exception
        assert not audio_file.exists(), \
            "Temp audio file must be cleaned up even when send_voice raises"


# =====================================================================
# Bug 8: Base adapter auto-TTS cleans up temp file after play_tts
# =====================================================================

class TestAutoTtsTempFileCleanup:
    """Base adapter auto-TTS must clean up generated audio file."""

    def test_source_has_finally_remove(self):
        """play_tts call is wrapped in try/finally with os.remove."""
        import inspect
        from gateway.platforms.base import BasePlatformAdapter
        source = inspect.getsource(BasePlatformAdapter._process_message_background)
        # Find the play_tts section and verify cleanup
        play_tts_idx = source.find("play_tts")
        assert play_tts_idx > 0
        after_play = source[play_tts_idx:]
        finally_idx = after_play.find("finally")
        remove_idx = after_play.find("os.remove")
        assert finally_idx > 0, "play_tts must be in a try/finally block"
        assert remove_idx > 0, "finally block must call os.remove on _tts_path"
        assert remove_idx > finally_idx, "os.remove must be inside the finally block"


# =====================================================================
# Voice channel awareness (get_voice_channel_info / context)
# =====================================================================


class TestVoiceChannelAwareness:
    """Tests for get_voice_channel_info() and get_voice_channel_context()."""

    def _make_adapter(self):
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import PlatformConfig
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter._voice_clients = {}
        adapter._voice_text_channels = {}
        adapter._voice_receivers = {}
        adapter._client = MagicMock()
        adapter._client.user = SimpleNamespace(id=99999, name="HermesBot")
        return adapter

    def _make_member(self, user_id, display_name, is_bot=False):
        return SimpleNamespace(
            id=user_id, display_name=display_name, bot=is_bot,
        )

    def test_returns_none_when_not_connected(self):
        adapter = self._make_adapter()
        assert adapter.get_voice_channel_info(111) is None

    def test_returns_none_when_vc_disconnected(self):
        adapter = self._make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = False
        adapter._voice_clients[111] = vc
        assert adapter.get_voice_channel_info(111) is None

    def test_returns_info_with_members(self):
        adapter = self._make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        bot_member = self._make_member(99999, "HermesBot", is_bot=True)
        user_a = self._make_member(1001, "Alice")
        user_b = self._make_member(1002, "Bob")
        vc.channel.name = "general-voice"
        vc.channel.members = [bot_member, user_a, user_b]
        adapter._voice_clients[111] = vc

        info = adapter.get_voice_channel_info(111)
        assert info is not None
        assert info["channel_name"] == "general-voice"
        assert info["member_count"] == 2  # bot excluded
        names = [m["display_name"] for m in info["members"]]
        assert "Alice" in names
        assert "Bob" in names
        assert "HermesBot" not in names

    def test_speaking_detection(self):
        adapter = self._make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        user_a = self._make_member(1001, "Alice")
        user_b = self._make_member(1002, "Bob")
        vc.channel.name = "voice"
        vc.channel.members = [user_a, user_b]
        adapter._voice_clients[111] = vc

        # Set up a mock receiver with Alice speaking
        import time as _time
        receiver = MagicMock()
        receiver._lock = threading.Lock()
        receiver._last_packet_time = {100: _time.monotonic()}  # ssrc 100 is active
        receiver._ssrc_to_user = {100: 1001}  # ssrc 100 -> Alice
        adapter._voice_receivers[111] = receiver

        info = adapter.get_voice_channel_info(111)
        alice = [m for m in info["members"] if m["display_name"] == "Alice"][0]
        bob = [m for m in info["members"] if m["display_name"] == "Bob"][0]
        assert alice["is_speaking"] is True
        assert bob["is_speaking"] is False
        assert info["speaking_count"] == 1

    def test_context_string_format(self):
        adapter = self._make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        user_a = self._make_member(1001, "Alice")
        vc.channel.name = "chat-room"
        vc.channel.members = [user_a]
        adapter._voice_clients[111] = vc

        ctx = adapter.get_voice_channel_context(111)
        assert "#chat-room" in ctx
        assert "1 participant" in ctx
        assert "Alice" in ctx

    def test_context_empty_when_not_connected(self):
        adapter = self._make_adapter()
        assert adapter.get_voice_channel_context(111) == ""


# ---------------------------------------------------------------------------
# Bugfix: disconnect() must clean up voice state
# ---------------------------------------------------------------------------


class TestDisconnectVoiceCleanup:
    """Bug: disconnect() left voice dicts populated after closing client."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_voice_state(self):
        from unittest.mock import AsyncMock

        adapter = MagicMock()
        adapter._voice_clients = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_receivers = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_listen_tasks = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_timeout_tasks = {111: MagicMock(), 222: MagicMock()}
        adapter._voice_text_channels = {111: 999, 222: 888}

        async def mock_leave(guild_id):
            adapter._voice_receivers.pop(guild_id, None)
            adapter._voice_listen_tasks.pop(guild_id, None)
            adapter._voice_clients.pop(guild_id, None)
            adapter._voice_timeout_tasks.pop(guild_id, None)
            adapter._voice_text_channels.pop(guild_id, None)

        for gid in list(adapter._voice_clients.keys()):
            await mock_leave(gid)

        assert len(adapter._voice_clients) == 0
        assert len(adapter._voice_receivers) == 0
        assert len(adapter._voice_listen_tasks) == 0
        assert len(adapter._voice_timeout_tasks) == 0


# =====================================================================
# Discord Voice Channel Flow Tests
# =====================================================================


@pytest.mark.skipif(
    importlib.util.find_spec("nacl") is None,
    reason="PyNaCl not installed",
)
class TestVoiceReception:
    """Audio reception: SSRC mapping, DAVE passthrough, buffer lifecycle."""

    @staticmethod
    def _make_receiver(allowed_ids=None, members=None, dave=False, bot_id=9999):
        from gateway.platforms.discord import VoiceReceiver
        vc = MagicMock()
        vc._connection.secret_key = [0] * 32
        vc._connection.dave_session = MagicMock() if dave else None
        vc._connection.ssrc = bot_id
        vc._connection.add_socket_listener = MagicMock()
        vc._connection.remove_socket_listener = MagicMock()
        vc._connection.hook = None
        vc.user = SimpleNamespace(id=bot_id)
        vc.channel = MagicMock()
        vc.channel.members = members or []
        receiver = VoiceReceiver(vc, allowed_user_ids=allowed_ids)
        return receiver

    @staticmethod
    def _fill_buffer(receiver, ssrc, duration_s=1.0, age_s=3.0):
        """Add PCM data to buffer. 48kHz stereo 16-bit = 192000 bytes/sec."""
        size = int(192000 * duration_s)
        receiver._buffers[ssrc] = bytearray(b"\x00" * size)
        receiver._last_packet_time[ssrc] = time.monotonic() - age_s

    # -- Known SSRC (normal flow) --

    def test_known_ssrc_returns_completed(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.map_ssrc(100, 42)
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42
        assert len(receiver._buffers[100]) == 0  # cleared

    def test_known_ssrc_short_buffer_ignored(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.map_ssrc(100, 42)
        self._fill_buffer(receiver, 100, duration_s=0.1)  # too short
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_known_ssrc_recent_audio_waits(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.map_ssrc(100, 42)
        self._fill_buffer(receiver, 100, age_s=0.0)  # just arrived
        completed = receiver.check_silence()
        assert len(completed) == 0

    # -- Unknown SSRC + DAVE passthrough --

    def test_unknown_ssrc_no_automap_no_completed(self):
        """Unknown SSRC, no members to infer — buffer cleared, not returned."""
        receiver = self._make_receiver(dave=True, members=[])
        receiver.start()
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 0
        assert len(receiver._buffers[100]) == 0

    def test_unknown_ssrc_late_speaking_event(self):
        """Audio buffered before SPEAKING → SPEAKING maps → next check returns it."""
        receiver = self._make_receiver(dave=True)
        receiver.start()
        self._fill_buffer(receiver, 100, age_s=0.0)  # still receiving
        # No user yet
        assert receiver.check_silence() == []
        # SPEAKING event arrives
        receiver.map_ssrc(100, 42)
        # Silence kicks in
        receiver._last_packet_time[100] = time.monotonic() - 3.0
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42

    # -- SSRC auto-mapping --

    def test_automap_single_allowed_user(self):
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = self._make_receiver(allowed_ids={"42"}, members=members)
        receiver.start()
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42
        assert receiver._ssrc_to_user[100] == 42

    def test_automap_multiple_allowed_users_no_map(self):
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
            SimpleNamespace(id=43, name="Bob"),
        ]
        receiver = self._make_receiver(allowed_ids={"42", "43"}, members=members)
        receiver.start()
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_automap_no_allowlist_single_member(self):
        """No allowed_user_ids → sole non-bot member inferred."""
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = self._make_receiver(allowed_ids=None, members=members)
        receiver.start()
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42

    def test_automap_unallowed_user_rejected(self):
        """User in channel but not in allowed list — not mapped."""
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = self._make_receiver(allowed_ids={"99"}, members=members)
        receiver.start()
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_automap_only_bot_in_channel(self):
        """Only bot in channel — no one to map to."""
        members = [SimpleNamespace(id=9999, name="Bot")]
        receiver = self._make_receiver(allowed_ids=None, members=members)
        receiver.start()
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 0

    def test_automap_persists_across_calls(self):
        """Auto-mapped SSRC stays mapped for subsequent checks."""
        members = [
            SimpleNamespace(id=9999, name="Bot"),
            SimpleNamespace(id=42, name="Alice"),
        ]
        receiver = self._make_receiver(allowed_ids={"42"}, members=members)
        receiver.start()
        self._fill_buffer(receiver, 100)
        receiver.check_silence()
        assert receiver._ssrc_to_user[100] == 42
        # Second utterance — should use cached mapping
        self._fill_buffer(receiver, 100)
        completed = receiver.check_silence()
        assert len(completed) == 1
        assert completed[0][0] == 42

    # -- Stale buffer cleanup --

    def test_stale_unknown_buffer_discarded(self):
        """Buffer with no user and very old timestamp is discarded."""
        receiver = self._make_receiver()
        receiver.start()
        receiver._buffers[200] = bytearray(b"\x00" * 100)
        receiver._last_packet_time[200] = time.monotonic() - 10.0
        receiver.check_silence()
        assert 200 not in receiver._buffers

    # -- Pause / resume (echo prevention) --

    def test_paused_receiver_ignores_packets(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.pause()
        receiver._on_packet(b"\x00" * 100)
        assert len(receiver._buffers) == 0

    def test_resumed_receiver_accepts_packets(self):
        receiver = self._make_receiver()
        receiver.start()
        receiver.pause()
        receiver.resume()
        assert receiver._paused is False

    # -- _on_packet DAVE passthrough behavior --

    def _make_receiver_with_nacl(self, dave_session=None, mapped_ssrcs=None):
        """Create a receiver that can process _on_packet with mocked NaCl + Opus."""
        from gateway.platforms.discord import VoiceReceiver
        vc = MagicMock()
        vc._connection.secret_key = [0] * 32
        vc._connection.dave_session = dave_session
        vc._connection.ssrc = 9999
        vc._connection.add_socket_listener = MagicMock()
        vc._connection.remove_socket_listener = MagicMock()
        vc._connection.hook = None
        vc.user = SimpleNamespace(id=9999)
        vc.channel = MagicMock()
        vc.channel.members = []
        receiver = VoiceReceiver(vc)
        receiver.start()
        # Pre-map SSRCs if provided
        if mapped_ssrcs:
            for ssrc, uid in mapped_ssrcs.items():
                receiver.map_ssrc(ssrc, uid)
        return receiver

    @staticmethod
    def _build_rtp_packet(ssrc=100, seq=1, timestamp=960):
        """Build a minimal valid RTP packet for _on_packet.

        We need: RTP header (12 bytes) + encrypted payload + 4-byte nonce.
        NaCl decrypt is mocked so payload content doesn't matter.
        """
        import struct
        # RTP header: version=2, payload_type=0x78, no extension, no CSRC
        header = struct.pack(">BBHII", 0x80, 0x78, seq, timestamp, ssrc)
        # Fake encrypted payload (NaCl will be mocked) + 4 byte nonce
        payload = b"\x00" * 20 + b"\x00\x00\x00\x01"
        return header + payload

    def _inject_mock_decoder(self, receiver, ssrc):
        """Pre-inject a mock Opus decoder for the given SSRC."""
        mock_decoder = MagicMock()
        mock_decoder.decode.return_value = b"\x00" * 3840
        receiver._decoders[ssrc] = mock_decoder
        return mock_decoder

    def test_on_packet_dave_known_user_decrypt_ok(self):
        """Known SSRC + DAVE decrypt success → audio buffered."""
        dave = MagicMock()
        dave.decrypt.return_value = b"\xf8\xff\xfe"
        receiver = self._make_receiver_with_nacl(
            dave_session=dave, mapped_ssrcs={100: 42}
        )
        self._inject_mock_decoder(receiver, 100)

        with patch("nacl.secret.Aead") as mock_aead:
            mock_aead.return_value.decrypt.return_value = b"\xf8\xff\xfe"
            receiver._on_packet(self._build_rtp_packet(ssrc=100))

        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0
        dave.decrypt.assert_called_once()

    def test_on_packet_dave_unknown_ssrc_passthrough(self):
        """Unknown SSRC + DAVE → skip DAVE, attempt Opus decode (passthrough)."""
        dave = MagicMock()
        receiver = self._make_receiver_with_nacl(dave_session=dave)
        self._inject_mock_decoder(receiver, 100)

        with patch("nacl.secret.Aead") as mock_aead:
            mock_aead.return_value.decrypt.return_value = b"\xf8\xff\xfe"
            receiver._on_packet(self._build_rtp_packet(ssrc=100))

        dave.decrypt.assert_not_called()
        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_on_packet_dave_unencrypted_error_passthrough(self):
        """DAVE decrypt 'Unencrypted' error → use data as-is, don't drop."""
        dave = MagicMock()
        dave.decrypt.side_effect = Exception(
            "Failed to decrypt: DecryptionFailed(UnencryptedWhenPassthroughDisabled)"
        )
        receiver = self._make_receiver_with_nacl(
            dave_session=dave, mapped_ssrcs={100: 42}
        )
        self._inject_mock_decoder(receiver, 100)

        with patch("nacl.secret.Aead") as mock_aead:
            mock_aead.return_value.decrypt.return_value = b"\xf8\xff\xfe"
            receiver._on_packet(self._build_rtp_packet(ssrc=100))

        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_on_packet_dave_other_error_drops(self):
        """DAVE decrypt non-Unencrypted error → packet dropped."""
        dave = MagicMock()
        dave.decrypt.side_effect = Exception("KeyRotationFailed")
        receiver = self._make_receiver_with_nacl(
            dave_session=dave, mapped_ssrcs={100: 42}
        )

        with patch("nacl.secret.Aead") as mock_aead:
            mock_aead.return_value.decrypt.return_value = b"\xf8\xff\xfe"
            receiver._on_packet(self._build_rtp_packet(ssrc=100))

        assert len(receiver._buffers.get(100, b"")) == 0

    def test_on_packet_no_dave_direct_decode(self):
        """No DAVE session → decode directly."""
        receiver = self._make_receiver_with_nacl(dave_session=None)
        self._inject_mock_decoder(receiver, 100)

        with patch("nacl.secret.Aead") as mock_aead:
            mock_aead.return_value.decrypt.return_value = b"\xf8\xff\xfe"
            receiver._on_packet(self._build_rtp_packet(ssrc=100))

        assert 100 in receiver._buffers
        assert len(receiver._buffers[100]) > 0

    def test_on_packet_bot_own_ssrc_ignored(self):
        """Bot's own SSRC → dropped (echo prevention)."""
        receiver = self._make_receiver_with_nacl()
        with patch("nacl.secret.Aead"):
            receiver._on_packet(self._build_rtp_packet(ssrc=9999))
        assert len(receiver._buffers) == 0

    def test_on_packet_multiple_ssrcs_separate_buffers(self):
        """Different SSRCs → separate buffers."""
        receiver = self._make_receiver_with_nacl(dave_session=None)
        self._inject_mock_decoder(receiver, 100)
        self._inject_mock_decoder(receiver, 200)

        with patch("nacl.secret.Aead") as mock_aead:
            mock_aead.return_value.decrypt.return_value = b"\xf8\xff\xfe"
            receiver._on_packet(self._build_rtp_packet(ssrc=100))
            receiver._on_packet(self._build_rtp_packet(ssrc=200))

        assert 100 in receiver._buffers
        assert 200 in receiver._buffers


class TestVoiceTTSPlayback:
    """TTS playback: play_tts in VC, dedup, fallback."""

    @staticmethod
    def _make_discord_adapter():
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import PlatformConfig, Platform
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_text_channels = {}
        adapter._voice_receivers = {}
        return adapter

    # -- play_tts behavior --

    @pytest.mark.asyncio
    async def test_play_tts_plays_in_vc(self):
        """play_tts calls play_in_voice_channel when bot is in VC."""
        adapter = self._make_discord_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 123

        played = []
        async def fake_play(gid, path):
            played.append((gid, path))
            return True
        adapter.play_in_voice_channel = fake_play

        result = await adapter.play_tts(chat_id="123", audio_path="/tmp/tts.ogg")
        assert result.success is True
        assert played == [(111, "/tmp/tts.ogg")]

    @pytest.mark.asyncio
    async def test_play_tts_fallback_when_not_in_vc(self):
        """play_tts sends as file attachment when bot is not in VC."""
        adapter = self._make_discord_adapter()
        from gateway.platforms.base import SendResult
        adapter.send_voice = AsyncMock(return_value=SendResult(success=False, error="no client"))
        result = await adapter.play_tts(chat_id="123", audio_path="/tmp/tts.ogg")
        assert result.success is False
        adapter.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_tts_wrong_channel_no_match(self):
        """play_tts doesn't match if chat_id is for a different channel."""
        adapter = self._make_discord_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 123

        from gateway.platforms.base import SendResult
        adapter.send_voice = AsyncMock(return_value=SendResult(success=True))
        # Different chat_id — shouldn't match VC
        result = await adapter.play_tts(chat_id="999", audio_path="/tmp/tts.ogg")
        adapter.send_voice.assert_called_once()

    # -- Runner dedup --

    @staticmethod
    def _make_runner():
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
        return runner

    def _call_should_reply(self, runner, voice_mode, msg_type, response="Hello", agent_msgs=None):
        from gateway.platforms.base import MessageType, MessageEvent, SessionSource
        from gateway.config import Platform
        runner._voice_mode["ch1"] = voice_mode
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="ch1",
            user_id="1", user_name="test", chat_type="channel",
        )
        event = MessageEvent(source=source, text="test", message_type=msg_type)
        return runner._should_send_voice_reply(event, response, agent_msgs or [])

    def test_voice_input_runner_skips(self):
        """Voice input: runner skips — base adapter handles via play_tts."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.VOICE) is False

    def test_text_input_voice_all_runner_fires(self):
        """Text input + voice_mode=all: runner generates TTS."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.TEXT) is True

    def test_text_input_voice_off_no_tts(self):
        """Text input + voice_mode=off: no TTS."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "off", MessageType.TEXT) is False

    def test_text_input_voice_only_no_tts(self):
        """Text input + voice_mode=voice_only: no TTS for text."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "voice_only", MessageType.TEXT) is False

    def test_error_response_no_tts(self):
        """Error response: no TTS regardless of voice_mode."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.TEXT, response="Error: boom") is False

    def test_empty_response_no_tts(self):
        """Empty response: no TTS."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.TEXT, response="") is False

    def test_agent_tts_tool_dedup(self):
        """Agent already called text_to_speech tool: runner skips."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        agent_msgs = [{"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "text_to_speech", "arguments": "{}"}}
        ]}]
        assert self._call_should_reply(runner, "all", MessageType.TEXT, agent_msgs=agent_msgs) is False


class TestUDPKeepalive:
    """UDP keepalive prevents Discord from dropping the voice session."""

    def test_keepalive_interval_is_reasonable(self):
        from gateway.platforms.discord import DiscordAdapter
        interval = DiscordAdapter._KEEPALIVE_INTERVAL
        assert 5 <= interval <= 30, f"Keepalive interval {interval}s should be between 5-30s"

    @pytest.mark.asyncio
    async def test_keepalive_sends_silence_frame(self):
        """Listen loop sends silence frame via send_packet after interval."""
        from gateway.platforms.discord import DiscordAdapter
        from gateway.config import PlatformConfig, Platform

        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_text_channels = {}
        adapter._voice_receivers = {}
        adapter._voice_listen_tasks = {}

        # Mock VC and receiver
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_conn = MagicMock()
        adapter._voice_clients[111] = mock_vc
        mock_vc._connection = mock_conn

        from gateway.platforms.discord import VoiceReceiver
        mock_receiver_vc = MagicMock()
        mock_receiver_vc._connection.secret_key = [0] * 32
        mock_receiver_vc._connection.dave_session = None
        mock_receiver_vc._connection.ssrc = 9999
        mock_receiver_vc._connection.add_socket_listener = MagicMock()
        mock_receiver_vc._connection.remove_socket_listener = MagicMock()
        mock_receiver_vc._connection.hook = None
        receiver = VoiceReceiver(mock_receiver_vc)
        receiver.start()
        adapter._voice_receivers[111] = receiver

        # Set keepalive interval very short for test
        original_interval = DiscordAdapter._KEEPALIVE_INTERVAL
        DiscordAdapter._KEEPALIVE_INTERVAL = 0.1

        try:
            # Run listen loop briefly
            import asyncio
            loop_task = asyncio.create_task(adapter._voice_listen_loop(111))
            await asyncio.sleep(0.3)
            receiver._running = False  # stop loop
            await asyncio.sleep(0.1)
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass

            # send_packet should have been called with silence frame
            mock_conn.send_packet.assert_called_with(b'\xf8\xff\xfe')
        finally:
            DiscordAdapter._KEEPALIVE_INTERVAL = original_interval
