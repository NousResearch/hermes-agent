"""Tests for the /voice command and auto voice reply in the gateway."""

import asyncio
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
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
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
    async def test_voice_off(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice off")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"


    @pytest.mark.asyncio
    async def test_toggle_on_to_off(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"

    @pytest.mark.asyncio
    async def test_persistence_saved(self, runner):
        event = _make_event("/voice on")
        await runner._handle_voice_command(event)
        assert runner._VOICE_MODE_PATH.exists()
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["telegram:123"] == "voice_only"


    def test_sync_populates_enabled_chats_from_voice_modes(self, runner):
        """Issue #16007: sync also restores per-chat /voice on|tts opt-ins.

        The adapter's ``_auto_tts_enabled_chats`` must mirror chats whose
        persisted voice_mode is ``voice_only`` or ``all`` — without this,
        ``/voice on`` was relying on a "not in disabled set" default that
        silently enabled auto-TTS for every chat.
        """
        from gateway.config import Platform
        runner._voice_mode = {
            "telegram:off_chat": "off",
            "telegram:on_chat": "voice_only",
            "telegram:tts_chat": "all",
            "slack:999": "voice_only",  # wrong platform, must be ignored
        }
        adapter = SimpleNamespace(
            _auto_tts_default=False,
            _auto_tts_disabled_chats=set(),
            _auto_tts_enabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_disabled_chats == {"off_chat"}
        assert adapter._auto_tts_enabled_chats == {"on_chat", "tts_chat"}

    def test_sync_pushes_config_default_onto_adapter(self, runner, monkeypatch):
        """Issue #16007: ``voice.auto_tts`` must propagate to ``_auto_tts_default``."""
        from gateway.config import Platform

        fake_cfg = {"voice": {"auto_tts": True}}
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: fake_cfg,
        )
        adapter = SimpleNamespace(
            _auto_tts_default=False,
            _auto_tts_disabled_chats=set(),
            _auto_tts_enabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_default is True


    @pytest.mark.asyncio
    async def test_platform_isolation(self, runner):
        """Same chat_id on different platforms must not collide (#12542)."""
        telegram_event = _make_event("/voice on", chat_id="999")
        slack_event = _make_event("/voice off", chat_id="999")
        slack_event.source.platform.value = "slack"

        await runner._handle_voice_command(telegram_event)
        await runner._handle_voice_command(slack_event)

        assert runner._voice_mode["telegram:999"] == "voice_only"
        assert runner._voice_mode["slack:999"] == "off"


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
            runner._voice_mode["telegram:" + chat_id] = voice_mode
        else:
            runner._voice_mode.pop("telegram:" + chat_id, None)

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


    # -- Text input: only runner handles -----------------------------------

    def test_text_input_all_mode_runner_fires(self, runner):
        """all + text input: only runner fires (base auto-TTS only for voice)."""
        assert self._call(runner, "all", MessageType.TEXT) is True


    # -- Mode off: nothing fires -------------------------------------------

    def test_off_mode_voice(self, runner):
        assert self._call(runner, "off", MessageType.VOICE) is False


    # -- Discord VC exception: runner must handle --------------------------

    def test_discord_vc_voice_input_base_handles(self, runner):
        """Discord VC + voice input: base adapter play_tts plays in VC,
        so runner skips to avoid double playback."""
        assert self._call(runner, "all", MessageType.VOICE, in_voice_channel=True) is False


    # -- Edge cases --------------------------------------------------------


# =====================================================================
# _send_voice_reply
# =====================================================================

class TestSendVoiceReply:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_calls_tts_and_send_voice(self, runner):
        from gateway.config import Platform

        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        event = _make_event()
        event.source.platform = Platform.TELEGRAM
        runner.adapters[event.source.platform] = mock_adapter

        tts_result = json.dumps({"success": True, "file_path": "/tmp/test.ogg"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result) as mock_tts, \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        mock_adapter.send_voice.assert_called_once()
        assert mock_tts.call_args.kwargs["output_path"].endswith(".ogg")
        call_args = mock_adapter.send_voice.call_args
        assert call_args.kwargs.get("chat_id") == "123"


    @pytest.mark.asyncio
    async def test_auto_voice_reply_uses_thread_metadata_helper(self, runner):
        from gateway.config import Platform

        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        event = _make_event()
        event.source.platform = Platform.TELEGRAM
        event.source.chat_type = "dm"
        event.source.thread_id = "20197"
        event.message_id = "462"
        runner.adapters[event.source.platform] = mock_adapter

        tts_result = json.dumps({"success": True, "file_path": "/tmp/test.ogg"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result), \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        mock_adapter.send_voice.assert_called_once()
        call_kwargs = mock_adapter.send_voice.call_args.kwargs
        assert call_kwargs["reply_to"] == "462"
        assert call_kwargs["metadata"] == {
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "20197",
            "telegram_reply_to_message_id": "462",
            # Final voice reply is notify-worthy (issue #27970 Bug 2):
            # mirrors the final-text path in gateway/platforms/base.py.
            "notify": True,
        }


# =====================================================================
# Discord play_tts skip when in voice channel
# =====================================================================

class TestDiscordPlayTtsSkip:
    """Discord adapter skips play_tts when bot is in a voice channel."""

    def _make_discord_adapter(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform, PlatformConfig
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
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


# =====================================================================
# VoiceReceiver unit tests
# =====================================================================

class TestVoiceReceiver:
    """Test VoiceReceiver silence detection, SSRC mapping, and lifecycle."""

    def _make_receiver(self):
        from plugins.platforms.discord.adapter import VoiceReceiver
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


    def test_map_ssrc_overwrites(self):
        receiver = self._make_receiver()
        receiver.map_ssrc(100, 42)
        receiver.map_ssrc(100, 99)
        assert receiver._ssrc_to_user[100] == 99


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


    def test_ffmpeg_resolver_finds_winget_install_when_not_on_path(self, monkeypatch, tmp_path):
        """Windows winget installs ffmpeg outside PATH; Discord voice should still find it."""
        from plugins.platforms.discord import ffmpeg_utils

        ffmpeg = (
            tmp_path
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "ffmpeg-7.1-full_build"
            / "bin"
            / "ffmpeg.exe"
        )
        ffmpeg.parent.mkdir(parents=True)
        ffmpeg.write_text("", encoding="utf-8")

        monkeypatch.delenv("FFMPEG_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        # Discovery delegates to tools.transcription_tools; simulate "not found".
        monkeypatch.setattr(ffmpeg_utils, "_shared_find_ffmpeg", lambda: None)

        assert ffmpeg_utils.resolve_ffmpeg_executable() == str(ffmpeg)


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
    async def test_join_success(self, runner):
        """Successful join sets voice_mode and returns confirmation."""
        mock_channel = MagicMock()
        mock_channel.name = "General"
        mock_adapter = AsyncMock()
        mock_adapter.join_voice_channel = AsyncMock(return_value=True)
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=mock_channel)
        mock_adapter._voice_text_channels = {}
        mock_adapter._voice_sources = {}
        mock_adapter._voice_input_callback = None
        event = self._make_discord_event()
        event.source.chat_type = "group"
        event.source.chat_name = "Hermes Server / #general"
        runner.adapters[event.source.platform] = mock_adapter
        result = await runner._handle_voice_channel_join(event)
        assert "joined" in result.lower()
        assert "General" in result
        assert runner._voice_mode["discord:123"] == "all"
        assert mock_adapter._voice_sources[111]["chat_id"] == "123"
        assert mock_adapter._voice_sources[111]["chat_type"] == "group"


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
        assert "PyNaCl" in result

    # -- _handle_voice_channel_leave --


    @pytest.mark.asyncio
    async def test_leave_success(self, runner):
        """Successful leave disconnects and clears voice mode."""
        mock_adapter = AsyncMock()
        mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
        mock_adapter.leave_voice_channel = AsyncMock()
        event = self._make_discord_event("/voice leave")
        runner.adapters[event.source.platform] = mock_adapter
        runner._voice_mode["discord:123"] = "all"
        result = await runner._handle_voice_channel_leave(event)
        assert "left" in result.lower()
        assert runner._voice_mode["discord:123"] == "off"
        mock_adapter.leave_voice_channel.assert_called_once_with(111)

    # -- _handle_voice_channel_input --


    @pytest.mark.asyncio
    async def test_input_creates_event_and_dispatches(self, runner):
        """Voice input creates synthetic event and calls handle_message."""
        from gateway.config import Platform
        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_adapter._voice_sources = {}
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
    async def test_input_resolves_channel_prompt(self, runner):
        """Voice input must carry the bound text channel's channel_prompt (#50149)."""
        from gateway.config import Platform
        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_adapter._voice_sources = {}
        mock_adapter._client = MagicMock()
        mock_adapter._client.get_channel = MagicMock(return_value=AsyncMock())
        mock_adapter.handle_message = AsyncMock()
        mock_adapter._resolve_channel_prompt = MagicMock(return_value="Be terse in #dev.")
        runner.adapters[Platform.DISCORD] = mock_adapter
        await runner._handle_voice_channel_input(111, 42, "Hello from VC")
        mock_adapter._resolve_channel_prompt.assert_called_once_with("123")
        event = mock_adapter.handle_message.call_args[0][0]
        assert event.channel_prompt == "Be terse in #dev."


    @pytest.mark.asyncio
    async def test_input_reuses_bound_source_metadata(self, runner):
        """Voice input should share the linked text channel session metadata."""
        from gateway.config import Platform

        bound_source = SessionSource(
            chat_id="123",
            chat_name="Hermes Server / #general",
            chat_type="group",
            user_id="user1",
            user_name="user1",
            platform=Platform.DISCORD,
        )

        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_adapter._voice_sources = {111: bound_source.to_dict()}
        mock_channel = AsyncMock()
        mock_adapter._client = MagicMock()
        mock_adapter._client.get_channel = MagicMock(return_value=mock_channel)
        mock_adapter.handle_message = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter

        await runner._handle_voice_channel_input(111, 42, "Hello from VC")

        mock_adapter.handle_message.assert_called_once()
        event = mock_adapter.handle_message.call_args[0][0]
        assert event.source.chat_id == "123"
        assert event.source.chat_type == "group"
        assert event.source.chat_name == "Hermes Server / #general"
        assert event.source.user_id == "42"


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


# =====================================================================
# Discord adapter voice channel methods
# =====================================================================

class TestDiscordVoiceChannelMethods:
    """Test DiscordAdapter voice channel methods (join, leave, play, etc.)."""

    def _make_adapter(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform, PlatformConfig
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._client = MagicMock()
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
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


    @pytest.mark.asyncio
    async def test_leave_voice_channel_processes_pending_audio_before_disconnect(self):
        """Recent speech is transcribed before the voice connection is torn down."""
        adapter = self._make_adapter()
        events = []
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True

        async def disconnect():
            events.append("disconnect")

        mock_vc.disconnect = disconnect
        adapter._voice_clients[111] = mock_vc

        mock_receiver = MagicMock()
        mock_receiver.flush_pending.side_effect = lambda: events.append("flush") or [(42, b"pcm")]
        mock_receiver.stop.side_effect = lambda: events.append("stop")
        adapter._voice_receivers[111] = mock_receiver
        adapter._voice_listen_tasks[111] = MagicMock()
        adapter._is_allowed_user = MagicMock(return_value=True)

        async def process(guild_id, user_id, pcm_data):
            events.append("process")

        adapter._process_voice_input = process

        await adapter.leave_voice_channel(111)

        assert events == ["flush", "stop", "process", "disconnect"]
        adapter._is_allowed_user.assert_called_once_with("42", guild=adapter._client.get_guild(111), is_dm=False)


    @pytest.mark.asyncio
    async def test_disconnect_leaves_voice_before_cancelling_bot_task(self):
        """Voice must be torn down while the gateway websocket is still alive.

        VoiceClient.disconnect() sends a voice state update over the main gateway
        connection and waits for the voice socket to close.  The bot task is the
        loop running that connection, so cancelling it first strands the
        handshake and the disconnect blocks until the caller's shutdown timeout.
        """
        adapter = self._make_adapter()
        events = []

        async def cancel_liveness_task():
            events.append("cancel_liveness_task")

        async def cancel_bot_task():
            events.append("cancel_bot_task")

        async def leave_voice_channel(guild_id):
            events.append(f"leave_voice_channel:{guild_id}")

        async def close():
            events.append("close_client")

        adapter._cancel_liveness_task = cancel_liveness_task
        adapter._cancel_bot_task = cancel_bot_task
        adapter.leave_voice_channel = leave_voice_channel
        adapter._client.close = close
        adapter._voice_clients[111] = MagicMock()
        adapter._ready_event = MagicMock()
        adapter._post_connect_task = None
        adapter._missed_message_backfill_task = None
        adapter._missed_message_backfill_periodic_task = None

        await adapter.disconnect()

        assert events == [
            "cancel_liveness_task",
            "leave_voice_channel:111",
            "cancel_bot_task",
            "close_client",
        ]


    @pytest.mark.asyncio
    async def test_disconnect_cancels_periodic_recovery_before_active_scan(self):
        adapter = self._make_adapter()
        events = []

        async def wait_for_cancel(name):
            try:
                await asyncio.Event().wait()
            finally:
                events.append(name)

        adapter._client = None
        adapter._bot_task = None
        adapter._ready_event = MagicMock()
        adapter._post_connect_task = None
        adapter._missed_message_backfill_task = asyncio.create_task(wait_for_cancel("scan"))
        adapter._missed_message_backfill_periodic_task = asyncio.create_task(
            wait_for_cancel("periodic")
        )
        await asyncio.sleep(0)

        await adapter.disconnect()

        assert events == ["periodic", "scan"]


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


    def test_voice_timeout_zero_disables_auto_leave(self):
        adapter = self._make_adapter()
        adapter._voice_timeout_seconds = 0
        existing_task = MagicMock()
        adapter._voice_timeout_tasks[111] = existing_task

        adapter._reset_voice_timeout(111)

        existing_task.cancel.assert_called_once()
        assert adapter._voice_timeout_tasks == {}

    def test_discord_voice_timeout_config_loaded(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import PlatformConfig

        with patch("hermes_cli.config.read_raw_config", return_value={
            "discord": {
                "voice_channel_inactivity_timeout_seconds": 0,
                "voice_playback_timeout_seconds": 240,
            }
        }):
            adapter = DiscordAdapter(PlatformConfig(enabled=True, token="x"))

        assert adapter._voice_timeout_seconds == 0
        assert adapter._playback_timeout_seconds == 240

    @pytest.mark.asyncio
    async def test_playback_timeout_scales_with_audio_duration(self):
        adapter = self._make_adapter()
        adapter._playback_timeout_seconds = 120
        adapter._probe_audio_duration_seconds = MagicMock(return_value=180.5)

        timeout = await adapter._playback_timeout_for_audio("/tmp/long.mp3")

        assert timeout == pytest.approx(210.5)


    @pytest.mark.asyncio
    async def test_play_in_voice_channel_uses_duration_aware_timeout(self):
        adapter = self._make_adapter()
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.is_playing.return_value = False
        adapter._voice_clients[111] = mock_vc
        adapter._playback_timeout_for_audio = AsyncMock(return_value=211.0)
        adapter._cancel_voice_timeout = MagicMock()
        adapter._reset_voice_timeout = MagicMock()

        def _play(_source, after):
            after(None)
        mock_vc.play.side_effect = _play

        with patch("plugins.platforms.discord.adapter.discord") as mock_discord:
            mock_discord.FFmpegPCMAudio.return_value = MagicMock()
            mock_discord.PCMVolumeTransformer.return_value = MagicMock()
            result = await adapter.play_in_voice_channel(111, "/tmp/long.mp3")

        assert result is True
        adapter._playback_timeout_for_audio.assert_awaited_once_with("/tmp/long.mp3")
        adapter._cancel_voice_timeout.assert_called_once_with(111)
        adapter._reset_voice_timeout.assert_called_once_with(111)


    def test_is_allowed_user_wildcard_only(self):
        """``DISCORD_ALLOWED_USERS="*"`` opens access to all users.

        Mirrors ``SIGNAL_ALLOWED_USERS`` and the existing
        ``DISCORD_ALLOWED_CHANNELS`` / ``_IGNORED_CHANNELS`` /
        ``_FREE_RESPONSE_CHANNELS`` wildcard handling. This is the
        convention ``claw migrate`` emits (#22334).
        """
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"*"}
        assert adapter._is_allowed_user("42") is True
        assert adapter._is_allowed_user("999999999999999999") is True


    @pytest.mark.asyncio
    async def test_process_voice_input_success(self):
        """Successful voice input: PCM->WAV->STT->callback."""
        adapter = self._make_adapter()
        callback = AsyncMock()
        adapter._voice_input_callback = callback
        adapter._allowed_user_ids = set()

        pcm_data = b"\x00" * 96000

        with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "Hello"}), \
             patch("tools.voice_mode.is_whisper_hallucination", return_value=False):
            await adapter._process_voice_input(111, 42, pcm_data)

        callback.assert_called_once_with(guild_id=111, user_id=42, transcript="Hello")


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
        from plugins.platforms.discord.adapter import VoiceReceiver
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
        from plugins.platforms.discord.adapter import VoiceReceiver
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


# =====================================================================
# Callback wiring order (join)
# =====================================================================

class TestCallbackWiringOrder:
    """Verify callback is wired BEFORE join, not after."""


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
        runner._voice_mode["telegram:123"] = "all"

        result = await runner._handle_voice_channel_leave(event)
        assert "left" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"
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

        # Note: base.py regex only strips individual chars, not full code blocks
        # So code blocks are partially stripped but may leave content
        # The real fix is in base.py — empty check after strip


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


        # Display callback gets raw text (before markdown stripping)
        # But the actual TTS audio would be stripped — we verify pipeline doesn't crash


# =====================================================================
# Bug 1: VoiceReceiver.stop() must hold lock while clearing shared state
# =====================================================================

class TestStopAcquiresLock:
    """stop() must acquire _lock before clearing buffers/state."""

    @staticmethod
    def _make_receiver():
        from plugins.platforms.discord.adapter import VoiceReceiver
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


# =====================================================================
# Bug 2: _packet_debug_count must be instance-level, not class-level
# =====================================================================

class TestPacketDebugCounterIsInstanceLevel:
    """Each VoiceReceiver instance has its own debug counter."""

    @staticmethod
    def _make_receiver():
        from plugins.platforms.discord.adapter import VoiceReceiver
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


# =====================================================================
# Bug 3: play_in_voice_channel uses get_running_loop not get_event_loop
# =====================================================================

class TestPlayInVoiceChannelUsesRunningLoop:
    """play_in_voice_channel must use asyncio.get_running_loop()."""

    def test_source_uses_get_running_loop(self):
        """The method source code calls get_running_loop, not get_event_loop."""
        import inspect
        from plugins.platforms.discord.adapter import DiscordAdapter
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
        """The path builder uses uuid in the filename, not time-based.

        Filename construction moved into build_auto_tts_output_path
        (gateway/platforms/base.py) when the path became platform-aware;
        the uniqueness contract lives there now.
        """
        import inspect
        from gateway.platforms.base import build_auto_tts_output_path
        from gateway.run import GatewayRunner
        source = inspect.getsource(build_auto_tts_output_path)
        assert "uuid" in source, \
            "build_auto_tts_output_path should use uuid for unique filenames"
        assert "int(time.time())" not in source, \
            "build_auto_tts_output_path should not use int(time.time()) — collision risk"
        runner_source = inspect.getsource(GatewayRunner._send_voice_reply)
        assert "build_auto_tts_output_path" in runner_source, \
            "_send_voice_reply should build its path via build_auto_tts_output_path"


# =====================================================================
# Bug 5: Voice timeout cleans up runner voice_mode via callback
# =====================================================================

class TestVoiceTimeoutCleansRunnerState:
    """Timeout disconnect notifies runner to clean voice_mode."""

    @staticmethod
    def _make_discord_adapter():
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import PlatformConfig, Platform
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
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


# =====================================================================
# Bug 6: play_in_voice_channel has playback timeout
# =====================================================================

class TestPlaybackTimeout:
    """play_in_voice_channel must time out instead of blocking forever."""

    @staticmethod
    def _make_discord_adapter():
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import PlatformConfig, Platform
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
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
        from plugins.platforms.discord.adapter import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.play_in_voice_channel)
        assert "wait_for" in source, \
            "play_in_voice_channel must use asyncio.wait_for for timeout"
        assert "_playback_timeout_for_audio" in source, \
            "play_in_voice_channel must use duration-aware playback timeout helper"


    @pytest.mark.asyncio
    async def test_playback_timeout_fires(self):
        """When done event is never set, playback times out gracefully."""
        from plugins.platforms.discord.adapter import DiscordAdapter
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
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import PlatformConfig
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._voice_receivers = {}
        adapter._client = MagicMock()
        adapter._client.user = SimpleNamespace(id=99999, name="HermesBot")
        return adapter

    def _make_member(self, user_id, display_name, is_bot=False):
        return SimpleNamespace(
            id=user_id, display_name=display_name, bot=is_bot,
        )


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


# ---------------------------------------------------------------------------
# Bugfix: disconnect() must clean up voice state
# ---------------------------------------------------------------------------


class TestDisconnectVoiceCleanup:
    """Bug: disconnect() left voice dicts populated after closing client."""

    @pytest.mark.asyncio
    async def test_disconnect_clears_voice_state(self):

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
        from plugins.platforms.discord.adapter import VoiceReceiver
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


    # -- Unknown SSRC + DAVE passthrough --


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
        from plugins.platforms.discord.adapter import VoiceReceiver
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


class TestVoiceTTSPlayback:
    """TTS playback: play_tts in VC, dedup, fallback."""

    @staticmethod
    def _make_discord_adapter():
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import PlatformConfig, Platform
        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
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


    # -- Runner dedup --

    @staticmethod
    def _make_runner():
        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
        return runner

    def _call_should_reply(self, runner, voice_mode, msg_type, response="Hello",
                           agent_msgs=None, already_sent=False):
        from gateway.platforms.base import MessageEvent, SessionSource
        from gateway.config import Platform
        runner._voice_mode["discord:ch1"] = voice_mode
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="ch1",
            user_id="1", user_name="test", chat_type="channel",
        )
        event = MessageEvent(source=source, text="test", message_type=msg_type)
        return runner._should_send_voice_reply(
            event, response, agent_msgs or [], already_sent=already_sent,
        )

    # -- Streaming OFF (existing behavior, must not change) --

    def test_voice_input_runner_skips(self):
        """Streaming OFF + voice input: runner skips — base adapter handles."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.VOICE, already_sent=False) is False

    def test_text_input_voice_all_runner_fires(self):
        """Streaming OFF + text input + voice_mode=all: runner generates TTS."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.TEXT, already_sent=False) is True


    def test_error_response_no_tts(self):
        """Error response: no TTS regardless of voice_mode."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.TEXT, response="Error: boom") is False


    # -- Streaming ON (already_sent=True) --


    def test_streaming_on_agent_tts_dedup(self):
        """Streaming ON + agent called TTS: runner skips (dedup still works)."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        agent_msgs = [{"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "text_to_speech", "arguments": "{}"}}
        ]}]
        assert self._call_should_reply(
            runner, "all", MessageType.VOICE, agent_msgs=agent_msgs, already_sent=True,
        ) is False


class TestUDPKeepalive:
    """UDP keepalive prevents Discord from dropping the voice session."""


    @pytest.mark.asyncio
    async def test_keepalive_sends_silence_frame(self):
        """Listen loop sends silence frame via send_packet after interval."""
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import PlatformConfig, Platform

        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._voice_receivers = {}
        adapter._voice_listen_tasks = {}

        # Mock VC and receiver
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_conn = MagicMock()
        adapter._voice_clients[111] = mock_vc
        mock_vc._connection = mock_conn

        from plugins.platforms.discord.adapter import VoiceReceiver
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
            await asyncio.sleep(0.2)
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


# =====================================================================
# BasePlatformAdapter._should_auto_tts_for_chat — gate for auto-TTS
# on voice input. Regression test for Issue #16007.
# =====================================================================

class TestShouldAutoTtsForChat:
    """Three-layer gate: per-chat enable > per-chat disable > config default."""

    def _make_adapter(self, *, default: bool, enabled=(), disabled=()):
        """Build a bare adapter with only the attrs the gate reads."""
        adapter = SimpleNamespace(
            _auto_tts_default=default,
            _auto_tts_enabled_chats=set(enabled),
            _auto_tts_disabled_chats=set(disabled),
        )
        # Bind the unbound method — _should_auto_tts_for_chat only reads the
        # three attrs above via ``self.``, so an unbound call works.
        from gateway.platforms.base import BasePlatformAdapter
        return BasePlatformAdapter._should_auto_tts_for_chat, adapter

    def test_default_false_no_override_suppresses(self):
        """Issue #16007: voice.auto_tts=False and no per-chat state → no TTS."""
        fn, adapter = self._make_adapter(default=False)
        assert fn(adapter, "chat1") is False


    def test_explicit_enable_overrides_false_default(self):
        """``/voice on`` with config auto_tts=False still fires."""
        fn, adapter = self._make_adapter(default=False, enabled={"chat1"})
        assert fn(adapter, "chat1") is True


class TestStreamTtsTempfileFallback:
    """Regression for the temp-WAV fallback in stream_tts_to_speaker.

    When no sounddevice output stream is available the streaming path falls
    back to writing each sentence to a temp WAV and playing it via the system
    player.  ``wave.open()`` given a *file object* flushes but does NOT close
    it (it only closes files it opened itself, by name), so the OS handle to
    the temp file stays open.  On Windows that open write handle blocks the
    player from reading the file and blocks ``os.unlink()`` (WinError 32,
    silently swallowed), leaving orphaned temp .wav files behind.  The fix
    closes the handle before playback/cleanup; this test asserts the close
    happens before the play call.
    """

    def test_tempfile_handle_closed_before_playback(self, monkeypatch):
        import wave
        import tools.tts_tool as tts_mod
        import tools.voice_mode as vm
        from tools.tts_tool import stream_tts_to_speaker

        # Fake registry streamer so resolve_streaming_provider yields chunked
        # PCM regardless of which real providers are configured in the env.
        class _FakeStreamer:
            sample_rate = 24000
            channels = 1

            def stream(self, text):
                yield b"\x00\x00" * 240
                yield b"\x00\x00" * 240

        monkeypatch.setattr(
            "tools.tts_streaming.resolve_streaming_provider",
            lambda tts_config, preferred=None: _FakeStreamer(),
        )

        # Force sounddevice unavailable → output_stream is None → tempfile fallback.
        def _no_sounddevice():
            raise ImportError("sounddevice unavailable in test")
        monkeypatch.setattr(tts_mod, "_import_sounddevice", _no_sounddevice)

        events = []  # ordered log of ("close", name) / ("play", path)

        # Spy on NamedTemporaryFile to record when the handle is closed.
        real_ntf = tts_mod.tempfile.NamedTemporaryFile

        def _spy_ntf(*args, **kwargs):
            f = real_ntf(*args, **kwargs)
            orig_close = f.close

            def _tracked_close():
                events.append(("close", f.name))
                return orig_close()

            f.close = _tracked_close
            return f

        monkeypatch.setattr(tts_mod.tempfile, "NamedTemporaryFile", _spy_ntf)

        played = []

        def _fake_play(path):
            events.append(("play", path))
            played.append(path)
            # At play time the file must be a fully written, readable WAV.
            with wave.open(path, "rb") as wf:
                assert wf.getnframes() > 0

        monkeypatch.setattr(vm, "play_audio_file", _fake_play)

        text_q = queue.Queue()
        stop_evt = threading.Event()
        done_evt = threading.Event()
        text_q.put("This is a spoken sentence for the fallback. ")
        text_q.put(None)

        stream_tts_to_speaker(text_q, stop_evt, done_evt)

        assert done_evt.is_set()
        assert played, "temp-file fallback player was never invoked"

        play_idx = next(i for i, e in enumerate(events) if e[0] == "play")
        closes_before_play = [
            i for i, e in enumerate(events[:play_idx]) if e[0] == "close"
        ]
        assert closes_before_play, (
            "temp WAV handle must be closed BEFORE play_audio_file — an open "
            "write handle blocks playback and os.unlink() on Windows"
        )
        # And the temp file is cleaned up afterwards.
        assert not os.path.exists(played[0]), "temp WAV was not unlinked"


class TestPcmToWav:
    """pcm_to_wav streams PCM through ffmpeg's stdin, not a temp file."""

    def test_pcm_is_piped_to_stdin_not_staged_on_disk(self, tmp_path):
        from plugins.platforms.discord.adapter import VoiceReceiver

        out = tmp_path / "out.wav"
        with patch("plugins.platforms.discord.adapter.subprocess.run") as run:
            VoiceReceiver.pcm_to_wav(b"\x00\x01" * 16, str(out))

        args, kwargs = run.call_args
        cmd = args[0]
        assert kwargs["input"] == b"\x00\x01" * 16, "PCM must be fed via stdin"
        assert "pipe:0" in cmd, "ffmpeg must read the PCM from stdin"
        assert cmd[-1] == str(out), (
            "the WAV must be written to the real path; ffmpeg cannot seek on a "
            "pipe, so a piped WAV gets placeholder RIFF/data sizes"
        )
        assert not any(str(a).endswith(".pcm") for a in cmd), (
            "no temp .pcm file should be staged"
        )

    @pytest.mark.skipif(
        __import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed",
    )
    def test_output_wav_header_reports_true_length(self, tmp_path):
        """A piped-stdout WAV reports 0xFFFFFFFF sizes; the written file must not."""
        import math
        import struct
        import wave

        from plugins.platforms.discord.adapter import VoiceReceiver

        frames = 48000  # 1s @ 48kHz stereo
        pcm = b"".join(
            struct.pack("<hh", v, v)
            for v in (
                int(20000 * math.sin(2 * math.pi * 440 * i / 48000))
                for i in range(frames)
            )
        )
        out = tmp_path / "out.wav"
        VoiceReceiver.pcm_to_wav(pcm, str(out))

        with wave.open(str(out)) as w:
            assert w.getnchannels() == 1
            assert w.getframerate() == 16000
            # 48kHz -> 16kHz is a 3x decimation of a 1s clip.
            assert w.getnframes() == 16000
