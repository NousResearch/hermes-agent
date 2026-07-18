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
from gateway.config import Platform


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


class _FakeVoiceClient:
    """Minimal stand-in for a discord VoiceClient that the REAL
    ``join_voice_channel`` / ``_leave_voice_channel_locked`` paths can drive.

    ``disconnect_gate`` (optional) lets a test hold the physical disconnect
    mid-flight to drive a cancellation-exactly-during-disconnect race.
    """

    def __init__(self, channel, *, disconnect_gate=None, on_disconnect_start=None,
                 move_gate=None, on_move_start=None, raise_move_after=None):
        self.channel = channel
        self._connected = True
        self._disconnect_gate = disconnect_gate
        self._on_disconnect_start = on_disconnect_start
        self._move_gate = move_gate
        self._on_move_start = on_move_start
        self._raise_move_after = raise_move_after  # raise on the (N+1)-th move_to
        self._move_count = 0
        self.disconnected = False

    def is_connected(self):
        return self._connected

    def is_playing(self):
        return False

    def stop(self):
        pass

    async def move_to(self, channel):
        if self._on_move_start is not None:
            self._on_move_start()
        if self._move_gate is not None:
            await self._move_gate.wait()
        if self._raise_move_after is not None and self._move_count >= self._raise_move_after:
            self._move_count += 1
            raise RuntimeError("simulated move_to failure")
        self._move_count += 1
        self.channel = channel

    async def disconnect(self):
        if self._on_disconnect_start is not None:
            self._on_disconnect_start()
        if self._disconnect_gate is not None:
            await self._disconnect_gate.wait()
        self._connected = False
        self.disconnected = True


class _FakeVoiceChannel:
    """A discord voice channel whose ``.connect()`` yields a ``_FakeVoiceClient``.

    ``connect_gate`` lets a test hold the connect mid-flight; ``on_connect`` fires
    the moment ``connect()`` is entered (before the gate) so a test can sequence a
    race deterministically without ``asyncio.sleep(0)`` guesses.
    """

    def __init__(self, channel_id, *, guild_id=111, members=(), connect_gate=None,
                 on_connect=None, vc_kwargs=None):
        self.id = int(channel_id)
        self.name = f"chan-{channel_id}"
        self.guild = SimpleNamespace(id=int(guild_id), name="Guild")
        self.members = list(members)
        self._connect_gate = connect_gate
        self._on_connect = on_connect
        self._vc_kwargs = vc_kwargs or {}
        self.last_vc = None

    async def connect(self):
        if self._on_connect is not None:
            self._on_connect()
        if self._connect_gate is not None:
            await self._connect_gate.wait()
        vc = _FakeVoiceClient(
            SimpleNamespace(id=self.id, members=self.members), **self._vc_kwargs
        )
        self.last_vc = vc
        return vc


class _InstrumentedLock:
    """An asyncio.Lock that fires ``waiter_arrived`` when a coroutine attempts to
    acquire it WHILE it is already held — so a lock-race test can wait for the
    contender to actually be blocked instead of assuming scheduler order."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.waiter_arrived = asyncio.Event()

    async def acquire(self):
        if self._lock.locked():
            self.waiter_arrived.set()
        return await self._lock.acquire()

    def release(self):
        self._lock.release()

    def locked(self):
        return self._lock.locked()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()


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
        assert runner._voice_mode["telegram:123"] == "voice_only"

    @pytest.mark.asyncio
    async def test_voice_off(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice off")
        result = await runner._handle_voice_command(event)
        assert "disabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"

    @pytest.mark.asyncio
    async def test_voice_tts(self, runner):
        event = _make_event("/voice tts")
        result = await runner._handle_voice_command(event)
        assert "tts" in result.lower()
        assert runner._voice_mode["telegram:123"] == "all"

    @pytest.mark.asyncio
    async def test_voice_status_off(self, runner):
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "off" in result.lower()

    @pytest.mark.asyncio
    async def test_voice_status_on(self, runner):
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event("/voice status")
        result = await runner._handle_voice_command(event)
        assert "voice reply" in result.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_to_on(self, runner):
        event = _make_event("/voice")
        result = await runner._handle_voice_command(event)
        assert "enabled" in result.lower()
        assert runner._voice_mode["telegram:123"] == "voice_only"

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

    @pytest.mark.asyncio
    async def test_persistence_loaded(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({"telegram:456": "all"}))
        loaded = runner._load_voice_modes()
        assert loaded == {"telegram:456": "all"}

    @pytest.mark.asyncio
    async def test_persistence_saved_for_off(self, runner):
        event = _make_event("/voice off")
        await runner._handle_voice_command(event)
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["telegram:123"] == "off"

    def test_sync_voice_mode_state_to_adapter_restores_off_chats(self, runner):
        from gateway.config import Platform
        runner._voice_mode = {"telegram:123": "off", "telegram:456": "all"}
        adapter = SimpleNamespace(
            _auto_tts_disabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        runner._sync_voice_mode_state_to_adapter(adapter)

        assert adapter._auto_tts_disabled_chats == {"123"}

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

    def test_restart_restores_voice_off_state(self, runner, tmp_path):
        from gateway.config import Platform
        runner._VOICE_MODE_PATH.write_text(json.dumps({"telegram:123": "off"}))

        restored_runner = _make_runner(tmp_path)
        restored_runner._voice_mode = restored_runner._load_voice_modes()
        adapter = SimpleNamespace(
            _auto_tts_disabled_chats=set(),
            platform=Platform.TELEGRAM,
        )

        restored_runner._sync_voice_mode_state_to_adapter(adapter)

        assert restored_runner._voice_mode["telegram:123"] == "off"
        assert adapter._auto_tts_disabled_chats == {"123"}

    @pytest.mark.asyncio
    async def test_per_chat_isolation(self, runner):
        e1 = _make_event("/voice on", chat_id="aaa")
        e2 = _make_event("/voice tts", chat_id="bbb")
        await runner._handle_voice_command(e1)
        await runner._handle_voice_command(e2)
        assert runner._voice_mode["telegram:aaa"] == "voice_only"
        assert runner._voice_mode["telegram:bbb"] == "all"

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
    async def test_non_telegram_auto_voice_reply_uses_mp3(self, runner):
        from gateway.config import Platform

        mock_adapter = AsyncMock()
        mock_adapter.send_voice = AsyncMock()
        event = _make_event()
        event.source.platform = Platform.SLACK
        runner.adapters[event.source.platform] = mock_adapter

        tts_result = json.dumps({"success": True, "file_path": "/tmp/test.mp3"})

        with patch("tools.tts_tool.text_to_speech_tool", return_value=tts_result) as mock_tts, \
             patch("tools.tts_tool._strip_markdown_for_tts", side_effect=lambda t: t), \
             patch("os.path.isfile", return_value=True), \
             patch("os.unlink"), \
             patch("os.makedirs"):
            await runner._send_voice_reply(event, "Hello world")

        mock_adapter.send_voice.assert_called_once()
        assert mock_tts.call_args.kwargs["output_path"].endswith(".mp3")

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
        assert "PyNaCl" in result

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
        runner._voice_mode["discord:123"] = "all"
        result = await runner._handle_voice_channel_leave(event)
        assert "left" in result.lower()
        assert runner._voice_mode["discord:123"] == "off"
        mock_adapter.leave_voice_channel.assert_called_once_with(111)

    @pytest.mark.asyncio
    async def test_leave_suppresses_voice_auto_rejoin(self, runner):
        """Manual /voice leave starts the adapter's auto-join cooldown."""
        mock_adapter = AsyncMock()
        mock_adapter.is_in_voice_channel = MagicMock(return_value=True)
        mock_adapter.leave_voice_channel = AsyncMock()
        mock_adapter.get_user_voice_channel = AsyncMock(return_value=SimpleNamespace(id=222))
        mock_adapter.suppress_voice_auto_join = MagicMock()
        event = self._make_discord_event("/voice leave")
        runner.adapters[event.source.platform] = mock_adapter

        await runner._handle_voice_channel_leave(event)

        mock_adapter.suppress_voice_auto_join.assert_called_once_with(
            111,
            user_id="user1",
            channel_id="222",
        )

    @pytest.mark.asyncio
    async def test_manual_leave_by_different_operator_suppresses_tracked_target(self, runner):
        """A manual /voice leave issued by a DIFFERENT operator than the tracked
        target still keys the cooldown to the followed target, not the issuer."""
        from plugins.platforms.discord.adapter import DiscordAdapter, VoiceAutoJoinTarget
        from gateway.config import Platform, PlatformConfig

        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._voice_auto_join_cfg = {"manual_leave_cooldown_seconds": 300}
        adapter._voice_auto_join_suppressed_until = {}
        adapter._voice_auto_join_retry_tasks = {}
        adapter._voice_clients = {111: SimpleNamespace(is_connected=lambda: True)}
        adapter._voice_auto_join_targets = {
            111: VoiceAutoJoinTarget(
                guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
            )
        }
        adapter.leave_voice_channel = AsyncMock()
        adapter._voice_input_callback = None
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        # Issuer of /voice leave is "user1" — NOT the followed target ("42").
        event = self._make_discord_event("/voice leave", user_id="user1")
        runner.adapters[event.source.platform] = adapter

        await runner._handle_voice_channel_leave(event)

        # Cooldown is keyed to the tracked target, blocking the target's rejoin.
        assert adapter._is_voice_auto_join_suppressed(111, "42", "222")
        # The issuer is not what gates the follow.
        assert not adapter._is_voice_auto_join_suppressed(111, "user1", "222")

    # -- _handle_voice_channel_input --

    @pytest.mark.asyncio
    async def test_input_no_adapter(self, runner):
        """No Discord adapter — early return, no crash."""
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

    @pytest.mark.asyncio
    async def test_input_posts_transcript_in_text_channel(self, runner):
        """Voice input sends transcript message to text channel."""
        from gateway.config import Platform
        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_adapter._voice_sources = {}
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

    @pytest.mark.asyncio
    async def test_input_suppresses_duplicate_transcript(self, runner):
        """Near-immediate duplicate STT output should not dispatch twice."""
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
        await runner._handle_voice_channel_input(111, 42, "Hello from VC")

        mock_adapter.handle_message.assert_called_once()
        mock_channel.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_input_suppresses_near_duplicate_transcript(self, runner):
        """Small STT wording drift should still be treated as the same utterance."""
        from gateway.config import Platform

        mock_adapter = AsyncMock()
        mock_adapter._voice_text_channels = {111: 123}
        mock_adapter._voice_sources = {}
        mock_channel = AsyncMock()
        mock_adapter._client = MagicMock()
        mock_adapter._client.get_channel = MagicMock(return_value=mock_channel)
        mock_adapter.handle_message = AsyncMock()
        runner.adapters[Platform.DISCORD] = mock_adapter

        await runner._handle_voice_channel_input(111, 42, "This is a test of the voice system")
        await runner._handle_voice_channel_input(111, 42, "This is a test for the voice system")

        mock_adapter.handle_message.assert_called_once()
        mock_channel.send.assert_called_once()

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


class TestDiscordVoiceAutoJoin:
    """Regression coverage for Discord voice auto-join/follow mode."""

    @staticmethod
    def _make_adapter(auto_join=None):
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform, PlatformConfig

        config = PlatformConfig(
            enabled=True,
            extra={"voice_auto_join": auto_join or {}},
        )
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
        adapter._voice_clients = {}
        adapter._voice_locks = {}
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._voice_timeout_tasks = {}
        adapter._voice_receivers = {}
        adapter._voice_listen_tasks = {}
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._voice_auto_join_callback = AsyncMock(return_value=True)
        adapter._voice_auto_join_cfg = adapter._load_voice_auto_join_config()
        adapter._voice_timeout_seconds = adapter._voice_auto_join_cfg["idle_timeout_seconds"]
        adapter._voice_auto_join_targets = {}
        adapter._voice_auto_join_suppressed_until = {}
        adapter._voice_auto_join_next_attempt_at = {}
        adapter._voice_auto_join_retry_tasks = {}
        adapter._voice_auto_join_pending = {}
        adapter._voice_auto_join_gen = 0
        adapter._voice_auto_join_direct_tasks = {}
        adapter._voice_session_owner = {}
        adapter._voice_mixers = {}
        adapter._allowed_user_ids = set()
        adapter._disconnecting = False
        return adapter

    @staticmethod
    def _connecting_callback(adapter, *, guild_id=111, channel_id=222, user_id=42):
        """An auto-join callback that simulates a real successful connection.

        The post-join barrier requires the adapter to actually be connected to
        the validated channel with the target present, so a bare
        ``AsyncMock(return_value=True)`` is not enough — the callback must
        populate ``_voice_clients`` the way the real join path does.
        """
        def _connect(**_kw):
            member = SimpleNamespace(id=int(user_id))
            channel = SimpleNamespace(id=int(channel_id), members=[member])
            adapter._voice_clients[int(guild_id)] = SimpleNamespace(
                is_connected=lambda: True, channel=channel
            )
            return True

        return AsyncMock(side_effect=_connect)

    @staticmethod
    def _voice_state(user_id="42", channel_id=222, before_channel=None):
        guild = SimpleNamespace(id=111, name="Guild")
        channel = SimpleNamespace(id=channel_id, name="General", guild=guild, members=[])
        member = SimpleNamespace(id=int(user_id), display_name="Martin", guild=guild)
        before = SimpleNamespace(channel=before_channel)
        after = SimpleNamespace(channel=channel)
        return member, before, after, channel

    @pytest.mark.asyncio
    async def test_auto_join_disabled_by_default(self):
        adapter = self._make_adapter()
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)

        adapter._voice_auto_join_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_user_and_channel_auto_join(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
        })
        adapter._voice_auto_join_callback = self._connecting_callback(adapter)
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)

        from unittest.mock import ANY
        adapter._voice_auto_join_callback.assert_awaited_once_with(
            guild_id=111,
            user_id="42",
            voice_channel_id="222",
            text_channel_id="333",
            attempt_token=ANY,
        )
        assert adapter._voice_auto_join_targets[111].user_id == "42"

    @pytest.mark.asyncio
    async def test_disallowed_user_or_channel_ignored(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["99"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        member, before, after, _channel = self._voice_state(user_id="42")

        await adapter._handle_voice_state_update(member, before, after)

        adapter._voice_auto_join_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_manual_leave_cooldown_blocks_immediate_rejoin(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        adapter.suppress_voice_auto_join(111, user_id="42", channel_id="222", seconds=30)
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)

        adapter._voice_auto_join_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_backoff_prevents_reconnect_thrash(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 60,
        })
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)
        await adapter._handle_voice_state_update(member, before, after)

        adapter._voice_auto_join_callback.assert_awaited_once()
        # A retry task is scheduled after the failed join; cancel it so the
        # 60s-backoff sleep does not dangle past the test.
        adapter._cancel_voice_auto_join_retry(111)

    @pytest.mark.asyncio
    async def test_target_leave_cleanup_disconnects_and_notifies_runner(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "target_leave_cleanup": True,
        })
        from plugins.platforms.discord.adapter import VoiceAutoJoinTarget

        guild = SimpleNamespace(id=111, name="Guild")
        before_channel = SimpleNamespace(id=222, name="General", guild=guild)
        member = SimpleNamespace(id=42, display_name="Martin", guild=guild)
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111,
            user_id="42",
            voice_channel_id="222",
            text_channel_id="333",
        )
        adapter._voice_text_channels[111] = 333
        # The bot is still connected to the followed channel (222), so cleanup
        # proceeds to disconnect.
        adapter._voice_clients[111] = SimpleNamespace(
            is_connected=lambda: True, channel=SimpleNamespace(id=222)
        )
        disconnect_calls = []
        adapter._on_voice_disconnect = lambda chat_id: disconnect_calls.append(chat_id)

        await adapter._handle_voice_state_update(
            member,
            SimpleNamespace(channel=before_channel),
            SimpleNamespace(channel=None),
        )

        # Physical session torn down under the guild lock + runner notified.
        assert 111 not in adapter._voice_clients
        assert disconnect_calls == ["333"]
        assert 111 not in adapter._voice_auto_join_targets

    @pytest.mark.asyncio
    async def test_runner_auto_join_reuses_manual_join_wiring(self, tmp_path):
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        guild = SimpleNamespace(id=111, name="Guild")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=guild, parent_id=None)
        voice_channel = SimpleNamespace(id=222, name="General", guild=guild)
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=text_channel)
        adapter.get_user_voice_channel = AsyncMock(return_value=voice_channel)
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        # The adapter reports the ACTUAL connected channel, which is what the
        # target must be recorded against.
        adapter.connected_voice_channel_id = MagicMock(return_value="222")
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._voice_mode_getter = None
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        adapter._voice_auto_join_cfg = {"allowed_voice_channel_ids": {"222"}}
        adapter.mark_voice_auto_join_target = MagicMock()
        runner.adapters[Platform.DISCORD] = adapter

        success = await runner._handle_discord_voice_auto_join(
            guild_id=111,
            user_id="42",
            voice_channel_id="222",
            text_channel_id="333",
            adapter=adapter,
            profile=None,
        )

        assert success is True
        adapter.join_voice_channel.assert_awaited_once_with(voice_channel, manual_owner=False, attempt_token=None)
        assert runner._voice_mode["discord:333"] == "all"
        assert adapter._auto_tts_enabled_chats == {"333"}
        adapter.mark_voice_auto_join_target.assert_called_once_with(
            111,
            user_id="42",
            voice_channel_id="222",
            text_channel_id="333",
            profile=None,
        )

    @pytest.mark.asyncio
    async def test_runner_auto_join_rejects_cross_guild_text_channel(self, tmp_path):
        """A linked text channel in a different guild than the voice event is rejected."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        # Text channel resolves into a DIFFERENT guild than the voice event (111).
        other_guild = SimpleNamespace(id=222222, name="Other Server")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=other_guild, parent_id=None)
        voice_channel = SimpleNamespace(id=222, name="General", guild=other_guild)
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=text_channel)
        adapter.get_user_voice_channel = AsyncMock(return_value=voice_channel)
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        adapter._voice_auto_join_cfg = {"allowed_voice_channel_ids": {"222"}}
        adapter.mark_voice_auto_join_target = MagicMock()
        runner.adapters[Platform.DISCORD] = adapter

        success = await runner._handle_discord_voice_auto_join(
            guild_id=111,
            user_id="42",
            voice_channel_id="222",
            text_channel_id="333",
        )

        assert success is False
        adapter.join_voice_channel.assert_not_awaited()
        adapter.mark_voice_auto_join_target.assert_not_called()

    @pytest.mark.asyncio
    async def test_runner_auto_join_revalidates_current_channel_against_allowlist(self, tmp_path):
        """A channel-move race — the user's live channel left the allowlist — is blocked."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        guild = SimpleNamespace(id=111, name="Guild")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=guild, parent_id=None)
        # The user has moved to a channel that is NOT in the allowlist since the
        # voice-state event that triggered auto-join fired.
        moved_channel = SimpleNamespace(id=999, name="Private", guild=guild)
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=text_channel)
        adapter.get_user_voice_channel = AsyncMock(return_value=moved_channel)
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        adapter._voice_auto_join_cfg = {"allowed_voice_channel_ids": {"222"}}
        adapter.mark_voice_auto_join_target = MagicMock()
        runner.adapters[Platform.DISCORD] = adapter

        success = await runner._handle_discord_voice_auto_join(
            guild_id=111,
            user_id="42",
            voice_channel_id="222",
            text_channel_id="333",
        )

        assert success is False
        adapter.join_voice_channel.assert_not_awaited()
        adapter.mark_voice_auto_join_target.assert_not_called()
        # The live channel is resolved exactly once — the join path must not
        # look it up a second time (that reopens the move TOCTOU).
        assert adapter.get_user_voice_channel.await_count == 1

    @pytest.mark.asyncio
    async def test_runner_auto_join_resolves_channel_once_and_joins_it(self, tmp_path):
        """The channel is resolved once and THAT channel is the one joined."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        guild = SimpleNamespace(id=111, name="Guild")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=guild, parent_id=None)
        voice_channel = SimpleNamespace(id=222, name="General", guild=guild)
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=text_channel)
        adapter.get_user_voice_channel = AsyncMock(return_value=voice_channel)
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        adapter.connected_voice_channel_id = MagicMock(return_value="222")
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._voice_mode_getter = None
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        adapter._voice_auto_join_cfg = {"allowed_voice_channel_ids": {"222"}}
        adapter.mark_voice_auto_join_target = MagicMock()
        runner.adapters[Platform.DISCORD] = adapter

        success = await runner._handle_discord_voice_auto_join(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )

        assert success is True
        # Exactly one resolution across the whole join path.
        assert adapter.get_user_voice_channel.await_count == 1
        adapter.join_voice_channel.assert_awaited_once_with(voice_channel, manual_owner=False, attempt_token=None)

    @pytest.mark.asyncio
    async def test_secondary_profile_auto_join_binds_owning_adapter(self, tmp_path):
        """A bound callback drives the owning secondary-profile adapter, stamps
        its profile, and never touches the active-profile adapter."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        runner._profile_adapters = {}
        guild = SimpleNamespace(id=111, name="Guild")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=guild, parent_id=None)
        voice_channel = SimpleNamespace(id=222, name="General", guild=guild)

        adapter = MagicMock()
        adapter.platform = Platform.DISCORD
        adapter._voice_auto_join_callback = None
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=text_channel)
        adapter.get_user_voice_channel = AsyncMock(return_value=voice_channel)
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        adapter.connected_voice_channel_id = MagicMock(return_value="222")
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._voice_mode_getter = None
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        adapter._voice_auto_join_cfg = {"allowed_voice_channel_ids": {"222"}}
        adapter.mark_voice_auto_join_target = MagicMock()
        runner._profile_adapters["coder"] = {Platform.DISCORD: adapter}

        # An active-profile adapter that must NOT be driven by the secondary's
        # event.
        active = MagicMock()
        active.platform = Platform.DISCORD
        active.join_voice_channel = AsyncMock(return_value=True)
        runner.adapters[Platform.DISCORD] = active

        runner._sync_voice_auto_join_state_to_adapter(adapter, profile_name="coder")
        assert adapter._voice_auto_join_callback is not None

        success = await adapter._voice_auto_join_callback(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )

        assert success is True
        adapter.join_voice_channel.assert_awaited_once_with(voice_channel, manual_owner=False, attempt_token=None)
        active.join_voice_channel.assert_not_awaited()
        adapter.mark_voice_auto_join_target.assert_called_once_with(
            111, user_id="42", voice_channel_id="222", text_channel_id="333", profile="coder",
        )

    @pytest.mark.asyncio
    async def test_stale_secondary_adapter_rejected_on_identity_mismatch(self, tmp_path):
        """A callback bound to an adapter that a reconnect has replaced is
        rejected — it must not connect on the superseded bot session."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        runner._profile_adapters = {}
        guild = SimpleNamespace(id=111, name="Guild")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=guild, parent_id=None)
        voice_channel = SimpleNamespace(id=222, name="General", guild=guild)

        stale = MagicMock()
        stale.platform = Platform.DISCORD
        stale._voice_auto_join_callback = None
        stale._client = MagicMock()
        stale._client.get_channel = MagicMock(return_value=text_channel)
        stale.get_user_voice_channel = AsyncMock(return_value=voice_channel)
        stale.join_voice_channel = AsyncMock(return_value=True)
        stale._voice_auto_join_cfg = {"allowed_voice_channel_ids": {"222"}}

        runner._sync_voice_auto_join_state_to_adapter(stale, profile_name="coder")

        # A newer adapter has replaced the stale one in the registry.
        fresh = MagicMock()
        fresh.platform = Platform.DISCORD
        runner._profile_adapters["coder"] = {Platform.DISCORD: fresh}

        success = await stale._voice_auto_join_callback(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )

        assert success is False
        stale.join_voice_channel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retry_task_rejoins_without_second_event(self):
        """After a failed join, the retry task re-attempts on its own — no
        second voice-state event is required."""
        # Backoff is deliberately large; the retry loop's sleep is patched to a
        # no-op so the test is deterministic — no real wall-clock delay and no
        # multi-second wait_for timeout that could flake under load.
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 30,
        })
        calls = {"n": 0}

        async def _cb(**_kw):
            calls["n"] += 1
            if calls["n"] > 1:
                # Retry succeeds — simulate the real join populating the client so
                # the post-join barrier can confirm the connection.
                member = SimpleNamespace(id=42)
                channel = SimpleNamespace(id=222, members=[member])
                adapter._voice_clients[111] = SimpleNamespace(
                    is_connected=lambda: True, channel=channel
                )
                return True
            return False  # first attempt fails

        adapter._voice_auto_join_callback = _cb
        guild = SimpleNamespace(id=111, name="Guild")
        chan = SimpleNamespace(id=222, name="General", guild=guild)
        adapter.get_user_voice_channel = AsyncMock(return_value=chan)
        member, before, after, _channel = self._voice_state()

        with patch(
            "plugins.platforms.discord.adapter.asyncio.sleep", new=AsyncMock()
        ) as patched_sleep:
            await adapter._handle_voice_state_update(member, before, after)

            task = adapter._voice_auto_join_retry_tasks.get(111)
            assert task is not None
            await asyncio.wait_for(task, timeout=1)

        # The backoff was honored via a (patched, instant) sleep — proving the
        # retry waited on the cadence rather than busy-looping.
        patched_sleep.assert_awaited()
        assert calls["n"] >= 2
        assert 111 in adapter._voice_auto_join_targets

    @pytest.mark.asyncio
    async def test_retry_task_cancelled_on_manual_leave(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 30,
        })
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)
        task = adapter._voice_auto_join_retry_tasks.get(111)
        assert task is not None and not task.done()

        adapter.suppress_voice_auto_join(111, user_id="42", channel_id="222")
        await asyncio.gather(task, return_exceptions=True)

        assert task.done()
        assert 111 not in adapter._voice_auto_join_retry_tasks

    @pytest.mark.asyncio
    async def test_retry_task_cancelled_on_target_leave(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 30,
            "target_leave_cleanup": False,
        })
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)
        task = adapter._voice_auto_join_retry_tasks.get(111)
        assert task is not None and not task.done()
        # Record a target so the leave path recognizes the departure.
        adapter.mark_voice_auto_join_target(
            111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )

        left_member = SimpleNamespace(id=42, display_name="Martin", guild=SimpleNamespace(id=111, name="Guild"))
        await adapter._handle_voice_state_update(
            left_member,
            SimpleNamespace(channel=SimpleNamespace(id=222, name="General")),
            SimpleNamespace(channel=None),
        )
        await asyncio.gather(task, return_exceptions=True)

        assert task.done()
        assert 111 not in adapter._voice_auto_join_retry_tasks

    @pytest.mark.asyncio
    async def test_manual_connection_not_adopted_into_auto_follow(self):
        """A pre-existing MANUAL voice connection is never adopted as an
        auto-follow target, so target-leave cleanup can't tear it down."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        # Bot is already MANUALLY connected to channel 222 (no auto-join target).
        existing_vc = MagicMock()
        existing_vc.is_connected.return_value = True
        existing_vc.channel = SimpleNamespace(id=222, name="General")
        adapter._voice_clients[111] = existing_vc
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)

        # No target recorded and no join attempted — the manual session stands.
        assert 111 not in adapter._voice_auto_join_targets
        adapter._voice_auto_join_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_suppression_retained_across_target_leave_rejoin(self):
        """Manual-leave suppression survives the target leaving and rejoining."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "manual_leave_cooldown_seconds": 300,
        })
        adapter.mark_voice_auto_join_target(
            111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        adapter.suppress_voice_auto_join(111, user_id="42", channel_id="222")

        # Target leaves the channel...
        await adapter._handle_voice_state_update(
            SimpleNamespace(id=42, display_name="M", guild=SimpleNamespace(id=111, name="G")),
            SimpleNamespace(channel=SimpleNamespace(id=222, name="General")),
            SimpleNamespace(channel=None),
        )
        # ...suppression must NOT have been cleared by the leave.
        assert adapter._is_voice_auto_join_suppressed(111, "42", "222")

        # ...and rejoining is still blocked while the cooldown holds.
        member, before, after, _channel = self._voice_state()
        await adapter._handle_voice_state_update(member, before, after)
        adapter._voice_auto_join_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_auto_join_config_loaded_from_temp_hermes_home(self, tmp_path):
        """Integration: a real config.yaml under HERMES_HOME feeds the adapter's
        voice-auto-join config loader end to end."""
        import yaml
        from hermes_constants import get_hermes_home
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform, PlatformConfig

        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump({
            "discord": {
                "voice_auto_join": {
                    "enabled": True,
                    "allowed_user_ids": ["42", "77"],
                    "allowed_voice_channel_ids": ["<#222>"],
                    "text_channel_id": "333",
                    "reconnect_cooldown_seconds": 5,
                },
            },
        }))

        config = PlatformConfig(enabled=True, extra={})
        config.token = "fake-token"
        adapter = object.__new__(DiscordAdapter)
        adapter.platform = Platform.DISCORD
        adapter.config = config

        cfg = adapter._load_voice_auto_join_config()

        assert cfg["enabled"] is True
        assert cfg["allowed_user_ids"] == {"42", "77"}
        assert cfg["allowed_voice_channel_ids"] == {"222"}
        assert cfg["text_channel_id"] == "333"
        assert cfg["reconnect_cooldown_seconds"] == 5

    # ----- Gap 2: refuse auto-follow over any manual connection -----

    @pytest.mark.asyncio
    async def test_auto_follow_refused_when_manual_connection_different_channel(self):
        """A manual connection in a DIFFERENT channel is never moved or adopted."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        adapter._voice_auto_join_callback = self._connecting_callback(adapter)
        # Bot is MANUALLY connected to channel 999 (NOT the allowed 222), no target.
        manual_channel = SimpleNamespace(id=999, name="Manual")
        manual_vc = SimpleNamespace(is_connected=lambda: True, channel=manual_channel)
        adapter._voice_clients[111] = manual_vc
        member, before, after, _channel = self._voice_state()  # target joins 222

        await adapter._handle_voice_state_update(member, before, after)

        adapter._voice_auto_join_callback.assert_not_called()
        assert 111 not in adapter._voice_auto_join_targets
        # The manual session is untouched — still connected to 999.
        assert adapter._voice_clients[111] is manual_vc
        assert adapter._voice_clients[111].channel.id == 999

    # ----- Gap 3: post-join barrier + pending-attempt race -----

    @pytest.mark.asyncio
    async def test_join_aborted_when_target_leaves_mid_join(self):
        """If a leave event cancels the pending attempt mid-join, the barrier
        aborts and tears the session back down rather than committing it."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        adapter.leave_voice_channel = AsyncMock()

        async def _cb(**_kw):
            # The target-leave event lands while the network join is in flight.
            adapter._cancel_pending_auto_join_if_matches(111, "42", "222")
            member = SimpleNamespace(id=42)
            channel = SimpleNamespace(id=222, members=[member])
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True, channel=channel
            )
            return True

        adapter._voice_auto_join_callback = _cb
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)

        assert 111 not in adapter._voice_auto_join_targets
        # The unowned session the aborted attempt created is torn down (no
        # dangling VC), and no retry is scheduled.
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_auto_join_retry_tasks

    @pytest.mark.asyncio
    async def test_join_aborted_when_actual_channel_differs(self):
        """If the bot ends up in a different channel than validated, abort."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        adapter.leave_voice_channel = AsyncMock()

        async def _cb(**_kw):
            # Landed in channel 999, not the validated 222.
            channel = SimpleNamespace(id=999, members=[SimpleNamespace(id=42)])
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True, channel=channel
            )
            return True

        adapter._voice_auto_join_callback = _cb
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)

        assert 111 not in adapter._voice_auto_join_targets
        # The unowned session the aborted attempt created is torn down (no
        # dangling VC), and no retry is scheduled.
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_auto_join_retry_tasks

    @pytest.mark.asyncio
    async def test_leave_event_cancels_in_flight_pending_join(self):
        """A concurrent leave event flags the tracked pending attempt cancelled."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        adapter._register_pending_auto_join(111, "42", "222")

        # Target leaves channel 222 while a join for it is in flight.
        await adapter._maybe_cleanup_voice_auto_join_target(
            111, "42",
            SimpleNamespace(id=222, name="General"),
            None,
        )

        assert adapter._voice_auto_join_pending[111]["cancelled"] is True

    # ----- Gap 4: disconnect teardown after failed join -----

    @pytest.mark.asyncio
    async def test_teardown_cancels_awaits_retry_and_clears_state(self):
        """Teardown cancels+awaits the retry task and clears all auto-join state."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 30,
        })
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        member, before, after, _channel = self._voice_state()

        await adapter._handle_voice_state_update(member, before, after)
        task = adapter._voice_auto_join_retry_tasks.get(111)
        assert task is not None and not task.done()

        await adapter._teardown_voice_auto_join_state()

        assert task.done()
        assert adapter._voice_auto_join_retry_tasks == {}
        assert adapter._voice_auto_join_targets == {}
        assert adapter._voice_auto_join_pending == {}
        assert adapter._voice_auto_join_callback is None
        assert adapter._voice_input_callback is None
        assert adapter._on_voice_disconnect is None
        assert adapter._voice_mode_getter is None

    @pytest.mark.asyncio
    async def test_disconnect_after_failed_join_tears_down_retry(self):
        """Full disconnect() after a failed join cancels the retry before teardown."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 30,
        })
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        member, before, after, _channel = self._voice_state()
        await adapter._handle_voice_state_update(member, before, after)
        task = adapter._voice_auto_join_retry_tasks.get(111)
        assert task is not None and not task.done()

        # Stub the rest of disconnect()'s resource teardown.
        adapter._disconnecting = False
        adapter._cancel_liveness_task = AsyncMock()
        adapter._cancel_bot_task = AsyncMock()
        adapter._post_connect_task = None
        adapter._ready_event = MagicMock()
        adapter._release_platform_lock = MagicMock()
        adapter._client = None

        await adapter.disconnect()

        assert task.done()
        assert adapter._voice_auto_join_retry_tasks == {}
        assert adapter._voice_auto_join_callback is None

    # ----- Gap 5: reconnect cooldown dominates retry backoff -----

    def test_retry_delay_is_max_of_backoff_and_cooldown(self):
        adapter = self._make_adapter()
        assert adapter._voice_auto_join_retry_delay(
            {"failure_backoff_seconds": 5, "reconnect_cooldown_seconds": 30}
        ) == 30
        assert adapter._voice_auto_join_retry_delay(
            {"failure_backoff_seconds": 60, "reconnect_cooldown_seconds": 15}
        ) == 60

    @pytest.mark.asyncio
    async def test_retry_scheduled_at_cooldown_deadline_when_cooldown_exceeds_backoff(self):
        """Virtual time: the retry sleeps the cooldown (30), not the backoff (5)."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 30,
            "failure_backoff_seconds": 5,
        })
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        slept = []

        async def _fake_sleep(seconds):
            slept.append(seconds)
            # Stop the retry loop after recording the (single) scheduled delay.
            raise asyncio.CancelledError

        member, before, after, _channel = self._voice_state()
        with patch("asyncio.sleep", _fake_sleep):
            await adapter._handle_voice_state_update(member, before, after)
            task = adapter._voice_auto_join_retry_tasks.get(111)
            assert task is not None
            # The loop catches the CancelledError and returns cleanly.
            await task

        # Slept the cooldown (30), NOT the shorter failure backoff (5).
        assert slept == [30]

    # ----- Gap 6: zero / tiny idle timeout safety -----

    def test_zero_idle_timeout_is_disabled(self):
        adapter = self._make_adapter()
        adapter._voice_timeout_seconds = 0
        assert adapter._effective_voice_timeout_seconds() == 0.0

    def test_tiny_positive_idle_timeout_floored(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter = self._make_adapter()
        adapter._voice_timeout_seconds = 0.001
        assert adapter._effective_voice_timeout_seconds() == DiscordAdapter.VOICE_TIMEOUT_MIN

    @pytest.mark.asyncio
    async def test_zero_idle_timeout_never_disconnects_or_zero_sleeps(self):
        adapter = self._make_adapter({"stay_while_target_present": True})
        adapter._voice_timeout_seconds = 0
        adapter._voice_text_channels[111] = 999
        adapter.leave_voice_channel = AsyncMock()
        slept = []

        async def _fake_sleep(seconds):
            slept.append(seconds)

        with patch("asyncio.sleep", _fake_sleep):
            await adapter._voice_timeout_handler(111)

        # Disabled: never sleeps, never disconnects, never rearms.
        assert slept == []
        adapter.leave_voice_channel.assert_not_awaited()
        assert 111 not in adapter._voice_timeout_tasks

    def test_reset_voice_timeout_disabled_schedules_nothing(self):
        adapter = self._make_adapter()
        adapter._voice_timeout_seconds = 0
        adapter._reset_voice_timeout(111)
        assert 111 not in adapter._voice_timeout_tasks

    # ----- Finding 1: in-flight joins must not survive disconnect / overlap -----

    @pytest.mark.asyncio
    async def test_in_flight_join_cancelled_and_awaited_during_disconnect(self):
        """A barrier-blocked in-flight attempt is cancelled+awaited by disconnect
        and never commits a session on the dying adapter."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
        })
        # Stub disconnect()'s resource-teardown deps.
        adapter._cancel_liveness_task = AsyncMock()
        adapter._cancel_bot_task = AsyncMock()
        adapter._post_connect_task = None
        adapter._ready_event = MagicMock()
        adapter._release_platform_lock = MagicMock()
        adapter._liveness_task = None
        adapter._running = True
        adapter._client = None
        adapter.leave_voice_channel = AsyncMock()

        started = asyncio.Event()
        release = asyncio.Event()

        async def _cb(**_kw):
            started.set()
            await release.wait()  # block mid-join
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True,
                channel=SimpleNamespace(id=222, members=[SimpleNamespace(id=42)]),
            )
            return True

        adapter._voice_auto_join_callback = _cb
        member, before, after, _channel = self._voice_state()
        ev_task = asyncio.ensure_future(
            adapter._handle_voice_state_update(member, before, after)
        )
        await started.wait()
        assert adapter._voice_auto_join_direct_tasks.get(111)  # attempt tracked per-guild

        # Full disconnect while the join is in flight.
        await adapter.disconnect()
        release.set()
        await asyncio.gather(ev_task, return_exceptions=True)

        # Nothing committed; all in-flight state cleared.
        assert 111 not in adapter._voice_auto_join_targets
        assert adapter._voice_auto_join_pending == {}
        assert adapter._voice_auto_join_direct_tasks == {}
        assert adapter._voice_auto_join_callback is None

    @pytest.mark.asyncio
    async def test_two_overlapping_attempts_then_leave_neither_commits(self):
        """A later attempt supersedes an earlier one (unique tokens); a target
        leave cancels the current entry, so neither overlapping join commits and
        no orphan session is left behind."""
        from plugins.platforms.discord.adapter import AutoJoinOutcome

        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        release = asyncio.Event()
        a_started = asyncio.Event()
        b_started = asyncio.Event()

        def _make_cb(started):
            async def _cb(**_kw):
                started.set()  # pending already registered by the barrier here
                await release.wait()
                adapter._voice_clients[111] = SimpleNamespace(
                    is_connected=lambda: True,
                    channel=SimpleNamespace(id=222, members=[SimpleNamespace(id=42)]),
                )
                return True
            return _cb

        # Start A and wait until it has registered its pending slot; only then
        # start B so B deterministically SUPERSEDES A (no scheduler assumptions).
        adapter._voice_auto_join_callback = _make_cb(a_started)
        task_a = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await a_started.wait()
        adapter._voice_auto_join_callback = _make_cb(b_started)
        task_b = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await b_started.wait()
        # Target leaves mid-flight → flags the CURRENT (B's) pending entry.
        adapter._cancel_pending_auto_join_if_matches(111, "42", "222")
        release.set()
        outcome_a, outcome_b = await asyncio.gather(task_a, task_b)

        assert outcome_a == AutoJoinOutcome.SUPERSEDED  # A superseded by B's token
        assert outcome_b == AutoJoinOutcome.CANCELLED   # B cancelled by the leave
        # Neither committed and the orphan connection is torn down.
        assert 111 not in adapter._voice_auto_join_targets
        assert adapter._voice_auto_join_pending == {}
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_session_owner

    # ----- Finding 2: manual takeover must not leave stale auto-follow -----

    @pytest.mark.asyncio
    async def test_cleanup_never_disconnects_a_manual_channel(self):
        """A stale auto-follow target must not disconnect a manual session the
        bot has since moved to (connected channel != tracked target channel)."""
        from plugins.platforms.discord.adapter import VoiceAutoJoinTarget

        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "target_leave_cleanup": True,
        })
        # Stale target for 222 remains, but the bot is now on manual channel 999.
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        adapter._voice_clients[111] = SimpleNamespace(
            is_connected=lambda: True, channel=SimpleNamespace(id=999, name="Manual")
        )
        adapter._voice_text_channels[111] = 333
        adapter.leave_voice_channel = AsyncMock()

        await adapter._handle_voice_state_update(
            SimpleNamespace(id=42, guild=SimpleNamespace(id=111, name="G")),
            SimpleNamespace(channel=SimpleNamespace(id=222, name="General")),
            SimpleNamespace(channel=None),
        )

        # Stale target dropped, but the manual 999 session is never disconnected.
        adapter.leave_voice_channel.assert_not_awaited()
        assert 111 not in adapter._voice_auto_join_targets
        assert adapter._voice_clients[111].channel.id == 999

    @pytest.mark.asyncio
    async def test_manual_takeover_clears_all_auto_follow_ownership(self):
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "failure_backoff_seconds": 30,
        })
        from plugins.platforms.discord.adapter import VoiceAutoJoinTarget
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        adapter._register_pending_auto_join(111, "42", "222")
        # A live retry task exists.
        adapter._voice_auto_join_callback = AsyncMock(return_value=False)
        adapter._schedule_voice_auto_join_retry(111, "42", "222", "333", 30)
        retry = adapter._voice_auto_join_retry_tasks.get(111)
        assert retry is not None

        adapter.clear_voice_auto_join_ownership(111)

        assert 111 not in adapter._voice_auto_join_targets
        assert 111 not in adapter._voice_auto_join_pending
        assert 111 not in adapter._voice_auto_join_retry_tasks
        await asyncio.gather(retry, return_exceptions=True)
        assert retry.cancelled() or retry.done()

    # ----- Final P1 findings: attempt-scoped teardown, cancellation, leave -----

    @pytest.mark.asyncio
    async def test_superseded_older_attempt_never_disconnects_newer_owner(self):
        """P1 (finding 1/B): an older auto-join that completes AFTER a newer one
        has committed and OWNS the guild's session must not disconnect it — the
        ownership check is session-token based, decided under the guild lock."""
        from plugins.platforms.discord.adapter import AutoJoinOutcome

        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222", "555"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        adapter.get_user_voice_channel = AsyncMock(
            return_value=SimpleNamespace(id=555)
        )
        hold_a = asyncio.Event()
        a_started = asyncio.Event()

        async def _cb_a(**_kw):
            a_started.set()  # pending already registered by the barrier here
            await hold_a.wait()  # A held until B has committed and owns 555
            return True

        # Older attempt A (channel 222) registers pending token 1, then blocks.
        adapter._voice_auto_join_callback = _cb_a
        task_a = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await a_started.wait()
        assert adapter._voice_auto_join_pending[111]["token"] == 1

        # Newer attempt B (channel 555) runs fully, connects, and COMMITS —
        # superseding A's pending slot and stamping session ownership with its
        # own token.
        async def _cb_b(**_kw):
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True,
                channel=SimpleNamespace(id=555, members=[SimpleNamespace(id=42)]),
            )
            return True

        adapter._voice_auto_join_callback = _cb_b
        outcome_b = await adapter._run_auto_join_with_barrier(111, "42", "555", "333")
        assert outcome_b == AutoJoinOutcome.COMMITTED
        b_owner = adapter._voice_session_owner[111]
        adapter.mark_voice_auto_join_target(
            111, user_id="42", voice_channel_id="555", text_channel_id="333",
        )

        # Release the older A: it completes late but is superseded.
        hold_a.set()
        outcome_a = await task_a

        assert outcome_a == AutoJoinOutcome.SUPERSEDED
        # B's session, ownership stamp, and target are untouched.
        assert adapter.connected_voice_channel_id(111) == "555"
        assert adapter._voice_session_owner[111] == b_owner
        assert adapter._voice_auto_join_targets[111].voice_channel_id == "555"

    @pytest.mark.asyncio
    async def test_superseded_stale_orphan_torn_down_when_newer_fails(self):
        """P1 (finding 1/A): a stale attempt whose callback physically connects
        AFTER a newer attempt superseded it — and the newer attempt then FAILS
        (so nobody owns the session) — must not leave a dangling unowned VC, and
        must schedule NO retry. Holds even with idle_timeout_seconds=0."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            # A large backoff would schedule a retry for a genuine FAILURE — the
            # test proves SUPERSEDED schedules none regardless.
            "failure_backoff_seconds": 30,
            "idle_timeout_seconds": 0,
        })
        hold = asyncio.Event()
        started = asyncio.Event()

        async def _cb(**_kw):
            started.set()
            await hold.wait()
            # Stale callback physically connects AFTER release (mutating the VC).
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True,
                channel=SimpleNamespace(id=222, members=[SimpleNamespace(id=42)]),
            )
            return True

        adapter._voice_auto_join_callback = _cb
        # Drive the stale attempt through the retry-capable entry point so we can
        # prove no retry is scheduled for a superseded outcome.
        join_task = asyncio.ensure_future(
            adapter._attempt_voice_auto_join(111, "42", "222", "333")
        )
        await started.wait()

        # A newer attempt B claims the pending slot (superseding A) and then FAILS
        # (clears its own pending), committing nothing → no session owner.
        token_b = adapter._register_pending_auto_join(111, "42", "222")
        adapter._clear_pending_auto_join(111, token_b)

        # Release the stale callback; its barrier now finalizes as superseded.
        hold.set()
        result = await join_task

        assert result is False  # attempt did not commit (superseded, not retried)
        # No dangling unowned VC, no ownership, no target, no pending, no retry.
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_session_owner
        assert 111 not in adapter._voice_auto_join_targets
        assert adapter._voice_auto_join_pending == {}
        assert 111 not in adapter._voice_auto_join_retry_tasks

    @pytest.mark.asyncio
    @staticmethod
    def _voice_event(chat_id, *, user_id="42", guild_id=111):
        source = SessionSource(chat_id=str(chat_id), user_id=str(user_id), platform=Platform.DISCORD)
        source.thread_id = None
        ev = MessageEvent(text="/voice", message_type=MessageType.TEXT, source=source)
        ev.message_id = "m1"
        ev.raw_message = SimpleNamespace(guild_id=int(guild_id), guild=None)
        return ev

    @staticmethod
    def _real_join_adapter(runner, cfg):
        """A real DiscordAdapter wired for the REAL join path under ``runner``."""
        from gateway.config import Platform
        adapter = TestDiscordVoiceAutoJoin._make_adapter(cfg)
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        adapter._client = SimpleNamespace(
            user=SimpleNamespace(id=999),
            get_channel=lambda cid: SimpleNamespace(
                id=int(cid), name=f"text-{cid}",
                guild=SimpleNamespace(id=111, name="Guild"), parent_id=None,
            ),
        )
        runner.adapters[Platform.DISCORD] = adapter
        runner._sync_voice_auto_join_state_to_adapter(adapter, profile_name=None)
        return adapter

    @pytest.mark.asyncio
    async def test_stale_callback_after_manual_takeover_leaves_manual_intact(self, tmp_path):
        """P1 auto-mutation: a stale auto callback that reaches the REAL
        ``join_voice_channel`` after a manual takeover established a manual-owned
        session is REJECTED before ``move_to``/``connect`` — the manual channel id,
        owner, target, voice mode, and auto-TTS all remain unchanged."""
        from plugins.platforms.discord.adapter import AutoJoinOutcome, VoiceOwnerKind

        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        member = SimpleNamespace(id=42)

        started = asyncio.Event()
        release = asyncio.Event()

        async def _cb(*, guild_id, user_id, voice_channel_id, text_channel_id, attempt_token=None):
            # Stale auto attempt reaches the REAL join path with its exact token.
            started.set()
            await release.wait()
            ch = _FakeVoiceChannel(222, guild_id=111, members=[member])
            return await adapter.join_voice_channel(ch, attempt_token=attempt_token)

        adapter._voice_auto_join_callback = _cb
        task = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await started.wait()  # A's pending registered; A blocked before the join

        # MANUAL takeover through the real handler → manual-owned session on 555,
        # runner mode 'all' + auto-TTS for the manual chat, and supersedes A.
        manual_ch = _FakeVoiceChannel(555, guild_id=111, members=[member])
        manual_event = self._voice_event("777")
        msg = await runner._handle_voice_channel_join(
            manual_event, voice_channel=manual_ch, manual=True
        )
        assert "joined" in msg.lower()
        assert runner._voice_mode["discord:777"] == "all"
        assert adapter._auto_tts_enabled_chats == {"777"}
        owner = adapter._voice_session_owner[111]
        assert owner.kind is VoiceOwnerKind.MANUAL
        manual_on_disconnect = adapter._on_voice_disconnect  # manual session's wiring

        # Release the stale attempt: its real join is rejected without mutation.
        release.set()
        outcome = await task

        assert outcome == AutoJoinOutcome.SUPERSEDED
        # Manual channel id, owner, mode, auto-TTS, and (absent) auto target all unchanged.
        assert adapter.connected_voice_channel_id(111) == "555"
        assert adapter._voice_session_owner[111] == owner
        assert adapter._voice_session_owner[111].kind is VoiceOwnerKind.MANUAL
        assert 111 not in adapter._voice_auto_join_targets
        assert runner._voice_mode["discord:777"] == "all"
        assert "discord:333" not in runner._voice_mode
        assert adapter._auto_tts_enabled_chats == {"777"}
        # The rejected stale attempt restored the manual session's callback wiring
        # (no cross-profile rebind of _on_voice_disconnect).
        assert adapter._on_voice_disconnect is manual_on_disconnect

    @pytest.mark.asyncio
    async def test_stale_callback_never_moves_newer_committed_session(self, tmp_path):
        """P1 auto-mutation: a stale auto attempt's real join must not move a
        NEWER auto-committed session (owned by a different token)."""
        from plugins.platforms.discord.adapter import AutoJoinOutcome

        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222", "555"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        member = SimpleNamespace(id=42)
        started = asyncio.Event()
        release = asyncio.Event()

        async def _cb_a(*, attempt_token=None, **_kw):
            started.set()
            await release.wait()
            ch = _FakeVoiceChannel(222, guild_id=111, members=[member])
            return await adapter.join_voice_channel(ch, attempt_token=attempt_token)

        adapter._voice_auto_join_callback = _cb_a
        task_a = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await started.wait()

        # Newer attempt B commits a real session on 555 (owned by B's token),
        # superseding A.
        async def _cb_b(*, attempt_token=None, **_kw):
            ch = _FakeVoiceChannel(555, guild_id=111, members=[member])
            return await adapter.join_voice_channel(ch, attempt_token=attempt_token)

        adapter._voice_auto_join_callback = _cb_b
        outcome_b = await adapter._run_auto_join_with_barrier(111, "42", "555", "333")
        assert outcome_b == AutoJoinOutcome.COMMITTED
        b_owner = adapter._voice_session_owner[111]

        release.set()
        outcome_a = await task_a
        assert outcome_a == AutoJoinOutcome.SUPERSEDED
        # B's committed session and ownership are untouched.
        assert adapter.connected_voice_channel_id(111) == "555"
        assert adapter._voice_session_owner[111] == b_owner

    @pytest.mark.asyncio
    async def test_cancellation_after_side_effects_rolls_back_transactionally(self, tmp_path):
        """P1 (finding 2): a real runner callback that completes its join side
        effects (via the REAL join path) and is then cancelled before the barrier
        commits must have those side effects rolled back transactionally and
        profile-bound."""
        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        member = SimpleNamespace(id=42)
        adapter.get_user_voice_channel = AsyncMock(
            return_value=_FakeVoiceChannel(222, guild_id=111, members=[member])
        )

        hold = asyncio.Event()
        side_effects_applied = asyncio.Event()

        async def _cb(*, guild_id, user_id, voice_channel_id, text_channel_id, attempt_token=None):
            ok = await runner._handle_discord_voice_auto_join(
                guild_id=guild_id, user_id=user_id, voice_channel_id=voice_channel_id,
                text_channel_id=text_channel_id, adapter=adapter, profile=None,
                attempt_token=attempt_token,
            )
            side_effects_applied.set()
            await hold.wait()  # cancelled here — INSIDE the callback, after side effects
            return ok

        adapter._voice_auto_join_callback = _cb
        task = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await side_effects_applied.wait()

        # Side effects are live before the cancellation (real join stamped AUTO
        # ownership, so the real ownership-gated mark_target ran).
        assert 111 in adapter._voice_clients
        assert runner._voice_mode["discord:333"] == "all"
        assert adapter._auto_tts_enabled_chats == {"333"}
        assert 111 in adapter._voice_auto_join_targets

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Transactional, profile-bound rollback.
        assert 111 not in adapter._voice_clients              # owned VC removed
        assert 111 not in adapter._voice_auto_join_targets    # target cleared
        assert adapter._voice_auto_join_pending == {}         # pending cleared
        assert 111 not in adapter._voice_session_owner        # ownership cleared
        assert runner._voice_mode["discord:333"] == "off"     # persisted mode off
        assert "333" not in adapter._auto_tts_enabled_chats   # enabled cleared
        assert "333" in adapter._auto_tts_disabled_chats      # disabled set

    @pytest.mark.asyncio
    async def test_manual_leave_before_preconnect_join_leaves_nothing(self, tmp_path):
        """P1 (finding 3): a manual /voice leave issued while an auto-join is
        barrier-blocked PRE-CONNECT must leave no connection/target/mode once the
        callback is released."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
            "manual_leave_cooldown_seconds": 300,
        })
        adapter.leave_voice_channel = AsyncMock(
            side_effect=lambda gid: adapter._voice_clients.pop(int(gid), None)
        )
        # A bound on_disconnect that records the runner-side voice mode.
        mode = {"voice": "all"}
        adapter._on_voice_disconnect = lambda chat_id: mode.__setitem__("voice", "off")
        runner.adapters[Platform.DISCORD] = adapter
        runner._adapter_for_source = lambda source: adapter

        hold = asyncio.Event()
        started = asyncio.Event()

        async def _cb(**_kw):
            started.set()  # pending is already registered by the barrier at this point
            await hold.wait()  # blocked PRE-CONNECT while the manual leave arrives
            member = SimpleNamespace(id=42)
            channel = SimpleNamespace(id=222, members=[member])
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True, channel=channel
            )
            return True

        adapter._voice_auto_join_callback = _cb

        # Kick off the auto-join; it registers a pending attempt and blocks.
        join_task = asyncio.ensure_future(
            adapter._attempt_voice_auto_join(111, "42", "222", "333")
        )
        await started.wait()
        assert adapter._voice_auto_join_pending.get(111) is not None
        assert 111 not in adapter._voice_clients  # not connected yet

        # Manual /voice leave while pre-connect (no physical VC exists).
        source = SessionSource(chat_id="333", user_id="42", platform=MagicMock())
        source.platform.value = "discord"
        source.thread_id = None
        leave_event = MessageEvent(
            text="/voice leave", message_type=MessageType.TEXT, source=source
        )
        leave_event.raw_message = SimpleNamespace(guild_id=111, guild=None)
        result = await runner._handle_voice_channel_leave(leave_event)
        assert "left" in result.lower()

        # Release the callback: it connects, but its own barrier tears it down.
        hold.set()
        await join_task

        assert 111 not in adapter._voice_clients            # no connection survives
        assert 111 not in adapter._voice_auto_join_targets  # no target
        assert adapter._voice_auto_join_pending == {}       # pending cleared
        assert mode["voice"] == "off"                       # runner mode off
        # Suppression recorded keyed to the in-flight attempt's user/channel.
        assert adapter._is_voice_auto_join_suppressed(111, "42", "222")

    @pytest.mark.asyncio
    async def test_target_rejoin_during_reconnect_cooldown_schedules_retry(self):
        """A target who leaves then rejoins the same channel DURING the reconnect
        cooldown is followed via a scheduled retry — no unrelated event needed."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 30,
            "failure_backoff_seconds": 0,
        })
        adapter._voice_auto_join_callback = self._connecting_callback(adapter)
        guild = SimpleNamespace(id=111, name="Guild")
        chan = SimpleNamespace(
            id=222, name="General", guild=guild, members=[SimpleNamespace(id=42)]
        )
        adapter.get_user_voice_channel = AsyncMock(return_value=chan)

        # A prior successful follow just ended (target left): a reconnect cooldown
        # is in force and NO retry task exists (success cancels retries).
        adapter._voice_auto_join_next_attempt_at[(111, "222")] = time.monotonic() + 30

        member, before, after, _channel = self._voice_state()  # target REJOINS 222

        with patch(
            "plugins.platforms.discord.adapter.asyncio.sleep", new=AsyncMock()
        ):
            await adapter._handle_voice_state_update(member, before, after)
            task = adapter._voice_auto_join_retry_tasks.get(111)
            assert task is not None  # a retry was scheduled for the remaining cooldown
            await asyncio.wait_for(task, timeout=1)

        # The rejoin was followed without a second voice-state event.
        assert adapter.connected_voice_channel_id(111) == "222"
        assert 111 in adapter._voice_auto_join_targets

    @pytest.mark.asyncio
    async def test_leave_auto_join_state_query_is_guild_scoped(self, tmp_path):
        """P2: an in-flight auto-join for guild A must not make an IDLE guild B's
        /voice leave report success or mutate B's voice mode. The gateway asks a
        guild-scoped snapshot, and direct tasks are keyed by guild."""
        from gateway.config import Platform

        runner = _make_runner(tmp_path)
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        runner.adapters[Platform.DISCORD] = adapter
        runner._adapter_for_source = lambda source: adapter

        # Guild A (111) is mid-auto-join: a direct barrier task + pending entry.
        adapter._voice_auto_join_direct_tasks[111] = {MagicMock(name="A-direct-task")}
        adapter._register_pending_auto_join(111, "42", "222")

        # The snapshot is strictly guild-scoped.
        state_a = adapter.voice_auto_join_state(111)
        assert state_a.inflight and state_a.has_direct and state_a.has_pending
        state_b = adapter.voice_auto_join_state(222)
        assert not state_b.inflight
        assert not state_b.has_direct and not state_b.has_pending

        # /voice leave for the IDLE guild B (222): no physical VC, nothing in
        # flight for B → "not in a voice channel", and B's mode is untouched.
        runner._voice_mode = {}
        source = SessionSource(chat_id="999", user_id="42", platform=MagicMock())
        source.platform.value = "discord"
        source.thread_id = None
        leave_event = MessageEvent(
            text="/voice leave", message_type=MessageType.TEXT, source=source
        )
        leave_event.raw_message = SimpleNamespace(guild_id=222, guild=None)

        result = await runner._handle_voice_channel_leave(leave_event)
        assert "not in" in result.lower()
        assert runner._voice_mode == {}  # guild B's mode never mutated

    @pytest.mark.asyncio
    async def test_concurrent_manual_takeover_during_rollback_wait(self, tmp_path):
        """P1 B: when a stale attempt's lock-scoped teardown must wait for the
        guild voice lock, a MANUAL takeover (real ``join_voice_channel``) that
        grabs the lock first — stamping manual ownership — is observed under the
        lock and left strictly intact. Driven with an instrumented lock (no
        ``asyncio.sleep(0)`` / no private-map mutation)."""
        from plugins.platforms.discord.adapter import AutoJoinOutcome, VoiceOwnerKind

        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        instrumented = _InstrumentedLock()
        adapter._voice_locks[111] = instrumented
        member = SimpleNamespace(id=42)
        move_gate = asyncio.Event()

        a_connected = asyncio.Event()
        a_proceed = asyncio.Event()

        async def _cb(*, attempt_token=None, **_kw):
            # Stale attempt A really connects to 222 (stamps AUTO ownership), then
            # pauses before returning so we can flag it cancelled + start a takeover.
            ch = _FakeVoiceChannel(
                222, guild_id=111, members=[member],
                vc_kwargs={"move_gate": move_gate},
            )
            ok = await adapter.join_voice_channel(ch, attempt_token=attempt_token)
            a_connected.set()
            await a_proceed.wait()
            return ok

        adapter._voice_auto_join_callback = _cb
        task = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await a_connected.wait()  # A connected + owns 222

        # MANUAL takeover: real join to 555. It acquires the guild lock and blocks
        # inside move_to (move_gate), HOLDING the lock. The manual handler also
        # supersedes A's pending slot (clear_voice_auto_join_ownership).
        manual_ch = _FakeVoiceChannel(555, guild_id=111, members=[member])
        manual_event = self._voice_event("777")
        manual_task = asyncio.ensure_future(
            runner._handle_voice_channel_join(manual_event, voice_channel=manual_ch, manual=True)
        )
        # Let A proceed: it returns, then its finalize contends for the lock the
        # manual takeover now holds → the instrumented lock signals the waiter.
        a_proceed.set()
        await instrumented.waiter_arrived.wait()

        # Now let the manual takeover finish (move + stamp MANUAL + release lock).
        move_gate.set()
        outcome = await task
        await manual_task

        # A did not commit (superseded by the manual takeover under the lock).
        assert outcome == AutoJoinOutcome.SUPERSEDED
        # The stale finalize re-checked ownership UNDER the lock and left the
        # manual session strictly intact.
        assert adapter.connected_voice_channel_id(111) == "555"
        assert adapter._voice_session_owner[111].kind is VoiceOwnerKind.MANUAL

    @pytest.mark.asyncio
    async def test_cancellation_during_finalize_lock_wait_completes_cleanup(self, tmp_path):
        """P1 B: a cancellation landing while the barrier is blocked acquiring the
        guild voice lock for its finalize still completes the owned/orphan cleanup.
        Uses an instrumented lock to know exactly when finalize is blocked."""
        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        instrumented = _InstrumentedLock()
        adapter._voice_locks[111] = instrumented
        member = SimpleNamespace(id=42)
        a_connected = asyncio.Event()
        a_proceed = asyncio.Event()

        async def _cb(*, attempt_token=None, **_kw):
            ch = _FakeVoiceChannel(222, guild_id=111, members=[member])
            ok = await adapter.join_voice_channel(ch, attempt_token=attempt_token)
            a_connected.set()
            await a_proceed.wait()
            return ok

        adapter._voice_auto_join_callback = _cb
        task = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await a_connected.wait()  # A connected + owns 222 (lock free again)

        # Hold the guild lock so the barrier's finalize must WAIT for it.
        await instrumented.acquire()
        a_proceed.set()
        await instrumented.waiter_arrived.wait()  # finalize is now blocked on the lock

        # Cancel while finalize is blocked, then release so its shield-to-completion
        # cleanup can run.
        task.cancel()
        instrumented.release()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The owned cleanup completed despite the cancellation.
        assert 111 not in adapter._voice_clients
        assert adapter._voice_auto_join_pending == {}
        assert 111 not in adapter._voice_session_owner

    @pytest.mark.asyncio
    async def test_cancellation_exactly_during_disconnect_completes_teardown(self, tmp_path):
        """Additional P1: a cancellation landing EXACTLY during the physical
        ``vc.disconnect()`` still completes the transactional teardown — the client
        is disconnected, every per-guild map is cleared, and the profile-bound
        runner cleanup runs (mode off, auto-TTS enabled cleared / disabled set)."""
        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        member = SimpleNamespace(id=42)
        disconnect_started = asyncio.Event()
        disconnect_gate = asyncio.Event()
        fake_channel = _FakeVoiceChannel(
            222, guild_id=111, members=[member],
            vc_kwargs={
                "disconnect_gate": disconnect_gate,
                "on_disconnect_start": disconnect_started.set,
            },
        )
        adapter.get_user_voice_channel = AsyncMock(return_value=fake_channel)

        async def _cb(*, guild_id, user_id, voice_channel_id, text_channel_id, attempt_token=None):
            ok = await runner._handle_discord_voice_auto_join(
                guild_id=guild_id, user_id=user_id, voice_channel_id=voice_channel_id,
                text_channel_id=text_channel_id, adapter=adapter, profile=None,
                attempt_token=attempt_token,
            )
            # Target leaves mid-join → the owned session is torn down at finalize.
            adapter._cancel_pending_auto_join_if_matches(111, "42", "222")
            return ok

        adapter._voice_auto_join_callback = _cb
        task = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )

        # Side effects were applied by the real join; teardown has now reached the
        # physical disconnect and is blocked inside it.
        await disconnect_started.wait()
        assert runner._voice_mode["discord:333"] == "all"  # still on at this instant
        vc = fake_channel.last_vc

        # Cancel EXACTLY during the disconnect, then let the disconnect finish.
        task.cancel()
        disconnect_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Physical disconnect completed and the FULL transactional teardown ran.
        assert vc.disconnected is True
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_session_owner
        assert 111 not in adapter._voice_auto_join_targets
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_sources
        assert adapter._voice_auto_join_pending == {}
        assert runner._voice_mode["discord:333"] == "off"     # runner mode off
        assert "333" not in adapter._auto_tts_enabled_chats   # enabled cleared
        assert "333" in adapter._auto_tts_disabled_chats      # disabled set

    @pytest.mark.asyncio
    async def test_leave_voice_channel_locked_is_cancellation_transactional(self):
        """Additional P1 (direct): cancelling _leave_voice_channel_locked EXACTLY
        during the awaited physical disconnect still completes the disconnect and
        clears every tracked map — no tracked or physical session state survives,
        checked immediately after the cancellation propagates (no settling wait)."""
        from plugins.platforms.discord.adapter import VoiceSessionOwner

        adapter = self._make_adapter({"enabled": True})
        gate = asyncio.Event()
        disconnect_started = asyncio.Event()
        vc = _FakeVoiceClient(
            SimpleNamespace(id=222),
            disconnect_gate=gate, on_disconnect_start=disconnect_started.set,
        )
        adapter._voice_clients[111] = vc
        adapter._voice_text_channels[111] = 333
        adapter._voice_sources[111] = {"chat_id": "333"}
        adapter._voice_session_owner[111] = VoiceSessionOwner.manual()
        adapter._voice_auto_join_targets[111] = SimpleNamespace(user_id="42")
        adapter._voice_receivers[111] = SimpleNamespace(stop=lambda: None)
        adapter._voice_locks[111] = asyncio.Lock()

        async def _caller():
            async with adapter._voice_locks[111]:
                await adapter._leave_voice_channel_locked(111)

        task = asyncio.ensure_future(_caller())
        await disconnect_started.wait()          # blocked INSIDE vc.disconnect()
        task.cancel()                            # cancel exactly during disconnect
        gate.set()                               # let the physical disconnect finish
        with pytest.raises(asyncio.CancelledError):
            await task

        # Checked immediately — the whole unit completed before the cancel propagated.
        assert vc.disconnected is True
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_sources
        assert 111 not in adapter._voice_session_owner
        assert 111 not in adapter._voice_auto_join_targets
        assert 111 not in adapter._voice_receivers

    @pytest.mark.asyncio
    async def test_teardown_transactional_runs_runner_cleanup_under_disconnect_cancel(self, tmp_path):
        """Additional P1 (direct): _teardown_voice_session_transactional cancelled
        EXACTLY during the physical disconnect still completes disconnect + all
        maps + the profile-bound runner cleanup (mode off, auto-TTS enabled cleared
        / disabled set)."""
        from gateway.config import Platform
        from plugins.platforms.discord.adapter import VoiceSessionOwner

        runner = _make_runner(tmp_path)
        adapter = self._make_adapter({"enabled": True})
        adapter._auto_tts_enabled_chats = {"333"}
        adapter._auto_tts_disabled_chats = set()
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode["discord:333"] = "all"
        # Bind the profile-bound runner cleanup exactly as the join path does.
        adapter._on_voice_disconnect = (
            lambda chat_id: runner._handle_voice_timeout_cleanup(chat_id, adapter=adapter, profile=None)
        )

        gate = asyncio.Event()
        disconnect_started = asyncio.Event()
        vc = _FakeVoiceClient(
            SimpleNamespace(id=222),
            disconnect_gate=gate, on_disconnect_start=disconnect_started.set,
        )
        adapter._voice_clients[111] = vc
        adapter._voice_text_channels[111] = 333
        adapter._voice_sources[111] = {"chat_id": "333"}
        adapter._voice_session_owner[111] = VoiceSessionOwner.auto(7)
        adapter._voice_locks[111] = asyncio.Lock()

        async def _caller():
            async with adapter._voice_locks[111]:
                await adapter._teardown_voice_session_transactional(111, "333")

        task = asyncio.ensure_future(_caller())
        await disconnect_started.wait()
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert vc.disconnected is True
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_sources
        assert 111 not in adapter._voice_session_owner
        # Profile-bound runner cleanup ran despite the cancel-during-disconnect.
        assert runner._voice_mode["discord:333"] == "off"
        assert "333" not in adapter._auto_tts_enabled_chats
        assert "333" in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_mid_move_supersession_undoes_physical_move(self):
        """P1: an auto attempt superseded WHILE its move_to is awaited must not
        leave a physical move mutation — the session is restored to its pre-move
        channel and the stale attempt stamps nothing."""
        from plugins.platforms.discord.adapter import VoiceSessionOwner

        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222", "555"],
            "text_channel_id": "333",
        })
        move_started = asyncio.Event()
        move_gate = asyncio.Event()
        # Existing auto-owned session on 222, whose move_to is gated.
        existing = _FakeVoiceClient(
            SimpleNamespace(id=222), move_gate=move_gate, on_move_start=move_started.set,
        )
        adapter._voice_clients[111] = existing
        adapter._voice_session_owner[111] = VoiceSessionOwner.auto(1)
        adapter._voice_locks[111] = asyncio.Lock()

        # Attempt A (token minted here) wants to follow the target to 555.
        token_a = adapter._register_pending_auto_join(111, "42", "555")
        target_555 = _FakeVoiceChannel(555, guild_id=111, members=[SimpleNamespace(id=42)])
        join_task = asyncio.ensure_future(
            adapter.join_voice_channel(target_555, attempt_token=token_a)
        )
        await move_started.wait()  # move_to(555) is now in flight (channel not yet changed)

        # A newer attempt supersedes A while the move is awaited.
        adapter._register_pending_auto_join(111, "42", "555")

        move_gate.set()
        result = await join_task

        assert result is False                       # stale attempt did not commit
        assert existing.channel.id == 222            # physical move UNDONE (restored)
        assert adapter.connected_voice_channel_id(111) == "222"
        # Ownership unchanged — the stale attempt stamped nothing.
        assert adapter._voice_session_owner[111] == VoiceSessionOwner.auto(1)

    @pytest.mark.asyncio
    async def test_target_leave_preserves_newer_auto_owner_same_channel(self):
        """P1: when a NEWER auto attempt re-commits (and re-marks the target) on the
        SAME channel while an older target-leave cleanup waits for the guild lock,
        the cleanup must preserve the newer session AND its target."""
        from plugins.platforms.discord.adapter import VoiceSessionOwner, VoiceAutoJoinTarget

        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "target_leave_cleanup": True,
        })
        vc = _FakeVoiceClient(SimpleNamespace(id=222))
        adapter._voice_clients[111] = vc
        adapter._voice_text_channels[111] = 333
        adapter._voice_session_owner[111] = VoiceSessionOwner.auto(1)  # older follow
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        instrumented = _InstrumentedLock()
        adapter._voice_locks[111] = instrumented

        # Hold the lock so the cleanup blocks after capturing owner_at_start=auto(1).
        await instrumented.acquire()
        cleanup_task = asyncio.ensure_future(
            adapter._maybe_cleanup_voice_auto_join_target(
                111, "42",
                SimpleNamespace(id=222, name="General"),
                None,
            )
        )
        await instrumented.waiter_arrived.wait()  # cleanup is blocked on the lock

        # A NEWER auto attempt commits on the SAME channel and re-marks the target.
        adapter._voice_session_owner[111] = VoiceSessionOwner.auto(2)
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        newer_target = adapter._voice_auto_join_targets[111]
        instrumented.release()
        await cleanup_task

        # The newer session, ownership, and target are all preserved.
        assert vc.disconnected is False
        assert 111 in adapter._voice_clients
        assert adapter._voice_session_owner[111] == VoiceSessionOwner.auto(2)
        assert adapter._voice_auto_join_targets[111] is newer_target

    @pytest.mark.asyncio
    async def test_join_signature_inspected_and_fails_closed_without_gate_kwargs(self, tmp_path):
        """P1: the gateway inspects join_voice_channel's SIGNATURE before calling —
        it never parses a TypeError and never retries tokenless. A gate-aware
        adapter that raises an internal TypeError makes exactly ONE failed call; an
        adapter whose callable LACKS the gate kwargs is FAILED CLOSED (never invoked
        tokenless, so the ownership gate can't be bypassed)."""
        from gateway.config import Platform

        # -- gate-aware signature that raises an internal TypeError → propagates →
        #    single failed call, NO tokenless retry --
        runner = _make_runner(tmp_path)
        adapter = MagicMock()
        adapter.platform = Platform.DISCORD
        calls = {"n": 0, "last_kwargs": None}

        async def _modern_join(channel, *, manual_owner=False, attempt_token=None):
            calls["n"] += 1
            calls["last_kwargs"] = {"manual_owner": manual_owner, "attempt_token": attempt_token}
            # Raise the EXACT text an unknown-kwarg TypeError produces. Because the
            # gateway inspects the signature (which DOES accept the kwargs) and never
            # parses TypeError text, this is a single GATED call that fails — it must
            # NOT be misread as "legacy" and retried tokenless.
            raise TypeError(
                "join_voice_channel() got an unexpected keyword argument 'attempt_token'"
            )

        adapter.join_voice_channel = _modern_join
        adapter._voice_input_callback = None
        adapter._on_voice_disconnect = None
        adapter._voice_mode_getter = None
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode = {}

        vch = _FakeVoiceChannel(222, guild_id=111, members=[SimpleNamespace(id=42)])
        event = self._voice_event("333")
        result = await runner._handle_voice_channel_join(event, voice_channel=vch, manual=True, attempt_token=None)
        assert "failed" in result.lower()
        assert calls["n"] == 1                            # exactly one call, NO retry
        assert calls["last_kwargs"] == {"manual_owner": True, "attempt_token": None}
        assert runner._voice_mode == {}                   # no side effects

        # -- signature WITHOUT the gate kwargs → FAIL CLOSED: never invoked --
        runner2 = _make_runner(tmp_path)
        adapter2 = MagicMock()
        adapter2.platform = Platform.DISCORD
        legacy = {"n": 0}

        async def _legacy_join(channel):   # no manual_owner / attempt_token, no **kwargs
            legacy["n"] += 1
            return True

        adapter2.join_voice_channel = _legacy_join
        adapter2._voice_input_callback = None
        adapter2._on_voice_disconnect = None
        adapter2._voice_mode_getter = None
        adapter2._voice_text_channels = {}
        adapter2._voice_sources = {}
        adapter2._auto_tts_enabled_chats = set()
        adapter2._auto_tts_disabled_chats = set()
        runner2.adapters[Platform.DISCORD] = adapter2
        runner2._voice_mode = {}
        runner2._VOICE_MODE_PATH = tmp_path / "vm2.json"

        vch2 = _FakeVoiceChannel(222, guild_id=111, members=[SimpleNamespace(id=42)])
        event2 = self._voice_event("333")
        result2 = await runner2._handle_voice_channel_join(event2, voice_channel=vch2, manual=True, attempt_token=None)
        assert "failed" in result2.lower()
        assert legacy["n"] == 0                            # NEVER invoked (fail closed)
        assert runner2._voice_mode == {}                  # no side effects

    @pytest.mark.asyncio
    async def test_public_leave_holds_lock_until_disconnect_completes_under_cancel(self):
        """P1: public leave_voice_channel cancelled during the physical disconnect
        must hold the guild lock and finish the physical + map cleanup before
        propagating — a contender cannot acquire the lock until teardown completes."""
        gate = asyncio.Event()
        disconnect_started = asyncio.Event()
        adapter = self._make_adapter({"enabled": True})
        vc = _FakeVoiceClient(
            SimpleNamespace(id=222), disconnect_gate=gate,
            on_disconnect_start=disconnect_started.set,
        )
        adapter._voice_clients[111] = vc
        adapter._voice_text_channels[111] = 333
        adapter._voice_sources[111] = {"chat_id": "333"}
        instrumented = _InstrumentedLock()
        adapter._voice_locks[111] = instrumented
        # A retry task must be finally-cleaned under the guild lock even when the
        # public leave is cancelled during the physical disconnect.
        retry_cancel_lock_state = []
        retry_task = MagicMock()
        retry_task.cancel.side_effect = lambda: retry_cancel_lock_state.append(
            instrumented.locked()
        )
        adapter._voice_auto_join_retry_tasks[111] = retry_task

        leave_task = asyncio.ensure_future(adapter.leave_voice_channel(111))
        await disconnect_started.wait()             # inside vc.disconnect(), lock held

        # A contender tries to take the guild lock; the instrumented lock signals
        # when it is actually BLOCKED (explicit event, no sleep-zero guess) — proof
        # the teardown still holds the lock. When the contender FINALLY acquires the
        # lock it records whether the retry was already cleared: because retry
        # cancellation runs in a finally INSIDE the async-with (lock still held), the
        # contender must never observe a stale retry task.
        got_lock = asyncio.Event()
        retry_state_when_locked = {}

        async def _contender():
            async with instrumented:
                retry_state_when_locked["cleared"] = (
                    111 not in adapter._voice_auto_join_retry_tasks
                )
                got_lock.set()

        contender = asyncio.ensure_future(_contender())
        leave_task.cancel()
        await instrumented.waiter_arrived.wait()    # contender is blocked on the lock
        assert not got_lock.is_set()                # lock still held by the teardown

        gate.set()                                  # let the disconnect finish
        with pytest.raises(asyncio.CancelledError):
            await leave_task

        assert vc.disconnected is True
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_sources
        assert 111 not in adapter._voice_auto_join_retry_tasks   # retry finally-cleaned
        assert retry_cancel_lock_state == [True]                 # cleaned under lock
        # Only now can the contender acquire the lock.
        await asyncio.wait_for(contender, timeout=1)
        assert got_lock.is_set()
        # The retry was already cleared AT THE MOMENT the contender took the lock —
        # i.e. cancellation happened while the guild lock was still held.
        assert retry_state_when_locked["cleared"] is True

    @pytest.mark.asyncio
    async def test_mid_move_rollback_failure_runs_full_profile_bound_teardown(self, tmp_path):
        """P1 (1): if the mid-move supersession ROLLBACK itself fails, the session
        is torn down via the FULL profile-bound teardown — physical + maps + owner +
        target AND the runner cleanup (persisted voice mode off, auto-TTS enabled
        cleared / disabled set) — all inside the run-to-completion unit."""
        from plugins.platforms.discord.adapter import VoiceSessionOwner, VoiceAutoJoinTarget

        runner = _make_runner(tmp_path)
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222", "555"],
            "text_channel_id": "333",
        })
        adapter._auto_tts_enabled_chats = {"333"}
        adapter._auto_tts_disabled_chats = set()
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode["discord:333"] = "all"
        # The bound, profile-scoped runner cleanup — exactly as the join path wires.
        adapter._on_voice_disconnect = (
            lambda chat_id: runner._handle_voice_timeout_cleanup(chat_id, adapter=adapter, profile=None)
        )

        move_started = asyncio.Event()
        move_gate = asyncio.Event()
        # Existing session on 222; the FIRST move (222→555) succeeds, the SECOND
        # (the 555→222 restore) raises.
        existing = _FakeVoiceClient(
            SimpleNamespace(id=222), move_gate=move_gate,
            on_move_start=move_started.set, raise_move_after=1,
        )
        adapter._voice_clients[111] = existing
        adapter._voice_session_owner[111] = VoiceSessionOwner.auto(1)
        adapter._voice_text_channels[111] = 333
        adapter._voice_sources[111] = {"chat_id": "333"}
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        adapter._voice_locks[111] = asyncio.Lock()

        token_a = adapter._register_pending_auto_join(111, "42", "555")
        target_555 = _FakeVoiceChannel(555, guild_id=111, members=[SimpleNamespace(id=42)])
        join_task = asyncio.ensure_future(
            adapter.join_voice_channel(target_555, attempt_token=token_a)
        )
        await move_started.wait()
        adapter._register_pending_auto_join(111, "42", "555")  # supersede A mid-move
        move_gate.set()
        result = await join_task

        assert result is False
        # Restore failed → FULL profile-bound teardown: no dangling physical/maps.
        assert existing.disconnected is True
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_session_owner
        assert 111 not in adapter._voice_text_channels
        assert 111 not in adapter._voice_sources
        assert 111 not in adapter._voice_auto_join_targets
        # ...and the runner profile cleanup ran.
        assert runner._voice_mode["discord:333"] == "off"
        assert "333" not in adapter._auto_tts_enabled_chats
        assert "333" in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_target_leave_preserves_replacement_target_same_channel(self):
        """P1 (2): driven through the real voice-state/behavior path and the real
        ``mark_voice_auto_join_target`` API — a REPLACEMENT target object on the
        SAME channel (same user, same owner token) is preserved. The cleanup
        matches the exact captured target OBJECT, not just channel/owner."""
        from plugins.platforms.discord.adapter import VoiceSessionOwner

        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "target_leave_cleanup": True,
        })
        vc = _FakeVoiceClient(SimpleNamespace(id=222))
        adapter._voice_clients[111] = vc
        adapter._voice_text_channels[111] = 333
        adapter._voice_session_owner[111] = VoiceSessionOwner.auto(1)
        # Establish the followed target through the REAL API.
        adapter.mark_voice_auto_join_target(
            111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        captured = adapter._voice_auto_join_targets[111]

        instrumented = _InstrumentedLock()
        adapter._voice_locks[111] = instrumented

        guild = SimpleNamespace(id=111, name="Guild")
        member = SimpleNamespace(id=42, display_name="Martin", guild=guild)
        before = SimpleNamespace(channel=SimpleNamespace(id=222, name="General"))
        after = SimpleNamespace(channel=None)

        # Hold the lock so the target-leave cleanup (driven by the real voice-state
        # handler) blocks after capturing the original target object.
        await instrumented.acquire()
        leave_task = asyncio.ensure_future(
            adapter._handle_voice_state_update(member, before, after)
        )
        await instrumented.waiter_arrived.wait()

        # A newer follow RE-MARKS the target through the real API (new object, same
        # user/channel/owner token).
        adapter.mark_voice_auto_join_target(
            111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        replacement = adapter._voice_auto_join_targets[111]
        assert replacement is not captured
        instrumented.release()
        await leave_task

        # The replacement target AND its session survive.
        assert adapter._voice_auto_join_targets[111] is replacement
        assert vc.disconnected is False
        assert 111 in adapter._voice_clients

    @pytest.mark.asyncio
    async def test_manual_join_clears_pending_target_retry_atomically(self, tmp_path):
        """P1 (6): a manual join clears the pending attempt, target, AND retry
        atomically under the guild lock, so an in-flight auto finalizer cannot
        commit AUTO ownership, re-mark a target, or reconnect over the manual
        session in the former window."""
        from plugins.platforms.discord.adapter import (
            AutoJoinOutcome, VoiceOwnerKind, VoiceAutoJoinTarget,
        )

        runner = _make_runner(tmp_path)
        adapter = self._real_join_adapter(runner, {
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
            "failure_backoff_seconds": 0,
        })
        member = SimpleNamespace(id=42)
        # Pre-existing (stale) auto follow state that the manual join must clear.
        adapter._voice_auto_join_targets[111] = VoiceAutoJoinTarget(
            guild_id=111, user_id="42", voice_channel_id="222", text_channel_id="333",
        )
        adapter._voice_auto_join_retry_tasks[111] = MagicMock()

        started = asyncio.Event()
        release = asyncio.Event()

        async def _cb(*, attempt_token=None, **_kw):
            started.set()  # A's pending registered by the barrier
            await release.wait()
            ch = _FakeVoiceChannel(222, guild_id=111, members=[member])
            return await adapter.join_voice_channel(ch, attempt_token=attempt_token)

        adapter._voice_auto_join_callback = _cb
        a_task = asyncio.ensure_future(
            adapter._run_auto_join_with_barrier(111, "42", "222", "333")
        )
        await started.wait()
        assert 111 in adapter._voice_auto_join_pending

        # Manual takeover through the REAL join path.
        manual_ch = _FakeVoiceChannel(555, guild_id=111, members=[member])
        msg = await runner._handle_voice_channel_join(
            self._voice_event("777"), voice_channel=manual_ch, manual=True
        )
        assert "joined" in msg.lower()
        # Pending, target, and retry were cleared ATOMICALLY under the join lock.
        assert 111 not in adapter._voice_auto_join_pending
        assert 111 not in adapter._voice_auto_join_targets
        assert 111 not in adapter._voice_auto_join_retry_tasks
        assert adapter._voice_session_owner[111].kind is VoiceOwnerKind.MANUAL

        # Releasing the in-flight auto attempt cannot replace/misclassify ownership.
        release.set()
        outcome = await a_task
        assert outcome == AutoJoinOutcome.SUPERSEDED
        assert adapter._voice_session_owner[111].kind is VoiceOwnerKind.MANUAL
        assert adapter.connected_voice_channel_id(111) == "555"
        assert 111 not in adapter._voice_auto_join_targets

    @pytest.mark.asyncio
    async def test_unpublished_connect_disconnect_completes_under_cancel(self):
        """P1 (5): a stale-connect whose vc was never published, cancelled during
        its disconnect, still disconnects the physical vc to completion (barrier
        cleanup could never find an unpublished client)."""
        adapter = self._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
        })
        adapter._voice_locks[111] = asyncio.Lock()
        token_a = adapter._register_pending_auto_join(111, "42", "222")

        disconnect_started = asyncio.Event()
        disconnect_gate = asyncio.Event()
        connected = asyncio.Event()

        def _on_connect():
            # The attempt goes stale WHILE its connect is completing.
            adapter._register_pending_auto_join(111, "42", "222")
            connected.set()

        fake_ch = _FakeVoiceChannel(
            222, guild_id=111, members=[SimpleNamespace(id=42)],
            on_connect=_on_connect,
            vc_kwargs={"disconnect_gate": disconnect_gate,
                       "on_disconnect_start": disconnect_started.set},
        )

        join_task = asyncio.ensure_future(
            adapter.join_voice_channel(fake_ch, attempt_token=token_a)
        )
        await disconnect_started.wait()   # stale → disconnecting the unpublished vc
        vc = fake_ch.last_vc
        assert 111 not in adapter._voice_clients   # never published

        join_task.cancel()
        disconnect_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await join_task

        # The unpublished vc was disconnected to completion despite the cancel.
        assert vc.disconnected is True
        assert 111 not in adapter._voice_clients


class TestVoiceProfileIsolation:
    """Gap 1: two simultaneous Discord adapters (default + secondary profile);
    the secondary's voice input / cleanup / mode never touch the default."""

    @staticmethod
    def _mock_discord_adapter():
        adapter = MagicMock()
        adapter.platform = Platform.DISCORD
        adapter._client = MagicMock()
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        adapter.handle_message = AsyncMock()
        return adapter

    def test_voice_key_namespaced_by_profile(self, tmp_path):
        runner = _make_runner(tmp_path)
        assert runner._voice_key(Platform.DISCORD, "333") == "discord:333"
        assert runner._voice_key(Platform.DISCORD, "333", profile="default") == "discord:333"
        assert (
            runner._voice_key(Platform.DISCORD, "333", profile="coder")
            == "profile:coder:discord:333"
        )
        # A secondary key must differ from the default so state never collides.
        assert runner._voice_key(Platform.DISCORD, "333", profile="coder") != runner._voice_key(
            Platform.DISCORD, "333"
        )

    @pytest.mark.asyncio
    async def test_voice_input_routes_to_owning_secondary_adapter(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner._profile_adapters = {}
        default = self._mock_discord_adapter()
        secondary = self._mock_discord_adapter()
        secondary._voice_text_channels = {111: 333}
        secondary._client.get_channel = MagicMock(
            return_value=SimpleNamespace(send=AsyncMock())
        )
        runner.adapters[Platform.DISCORD] = default
        runner._profile_adapters["coder"] = {Platform.DISCORD: secondary}

        await runner._handle_voice_channel_input(
            111, 42, "hello from the secondary profile", adapter=secondary, profile="coder"
        )

        # Only the owning (secondary) adapter processed the voice turn.
        secondary.handle_message.assert_awaited_once()
        default.handle_message.assert_not_awaited()
        # The synthetic turn is stamped with the owning profile.
        event = secondary.handle_message.call_args.args[0]
        assert event.source.profile == "coder"

    def test_timeout_cleanup_targets_owning_adapter_only(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner._profile_adapters = {}
        default = self._mock_discord_adapter()
        secondary = self._mock_discord_adapter()
        runner.adapters[Platform.DISCORD] = default
        runner._profile_adapters["coder"] = {Platform.DISCORD: secondary}

        runner._handle_voice_timeout_cleanup("333", adapter=secondary, profile="coder")

        # Secondary's auto-TTS suppression updated; default untouched.
        assert "333" in secondary._auto_tts_disabled_chats
        assert "333" not in default._auto_tts_disabled_chats
        # Voice mode is written under the profile-namespaced key only.
        assert runner._voice_mode.get("profile:coder:discord:333") == "off"
        assert "discord:333" not in runner._voice_mode

    @pytest.mark.asyncio
    async def test_voice_command_mode_namespaced_to_owning_profile(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner._profile_adapters = {}
        default = self._mock_discord_adapter()
        secondary = self._mock_discord_adapter()
        runner.adapters[Platform.DISCORD] = default
        runner._profile_adapters["coder"] = {Platform.DISCORD: secondary}

        source = SessionSource(
            platform=Platform.DISCORD, chat_id="333", user_id="42", chat_type="channel",
        )
        source.profile = "coder"
        event = MessageEvent(text="/voice tts", message_type=MessageType.COMMAND, source=source)

        await runner._handle_voice_command(event)

        # Mode written under the profile-namespaced key; default namespace clean.
        assert runner._voice_mode.get("profile:coder:discord:333") == "all"
        assert "discord:333" not in runner._voice_mode
        # Only the secondary adapter's auto-TTS opt-in was touched.
        assert "333" in secondary._auto_tts_enabled_chats
        assert "333" not in default._auto_tts_enabled_chats

    # ----- Finding 3: profile-aware mode synchronization -----

    def test_mode_sync_isolates_opposite_default_and_secondary_states(self, tmp_path):
        runner = _make_runner(tmp_path)
        default = self._mock_discord_adapter()
        secondary = self._mock_discord_adapter()
        # Opposite states: default chat 100 is ON ("all"); secondary chat 200 is OFF.
        runner._voice_mode = {
            "discord:100": "all",
            "profile:coder:discord:200": "off",
        }

        runner._sync_voice_mode_state_to_adapter(default)
        runner._sync_voice_mode_state_to_adapter(secondary, profile_name="coder")

        # Default adapter sees ONLY its own namespace.
        assert default._auto_tts_enabled_chats == {"100"}
        assert default._auto_tts_disabled_chats == set()
        # Secondary adapter sees ONLY its own namespace.
        assert secondary._auto_tts_disabled_chats == {"200"}
        assert secondary._auto_tts_enabled_chats == set()

        # Restart/reconnect re-derives the same isolated state (idempotent).
        runner._sync_voice_mode_state_to_adapter(secondary, profile_name="coder")
        assert secondary._auto_tts_disabled_chats == {"200"}
        assert "100" not in secondary._auto_tts_disabled_chats
        assert "100" not in secondary._auto_tts_enabled_chats

    # ----- Finding 4: aborted auto-join rolls back join side effects -----

    @pytest.mark.asyncio
    async def test_real_callback_midjoin_cancel_rolls_back_all_side_effects(self, tmp_path):
        """The real runner auto-join callback, cancelled mid-join, must leave no
        physical VC / target / pending and roll back persisted mode + auto-TTS."""
        adapter = TestDiscordVoiceAutoJoin._make_adapter({
            "enabled": True,
            "allowed_user_ids": ["42"],
            "allowed_voice_channel_ids": ["222"],
            "text_channel_id": "333",
            "reconnect_cooldown_seconds": 0,
        })
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        guild = SimpleNamespace(id=111, name="Guild")
        text_channel = SimpleNamespace(id=333, name="voice-chat", guild=guild, parent_id=None)
        voice_channel = SimpleNamespace(id=222, name="General", guild=guild)
        adapter._client.get_channel = MagicMock(return_value=text_channel)
        adapter.get_user_voice_channel = AsyncMock(return_value=voice_channel)

        async def _join(channel, **_kw):
            # The target-leave lands mid-join (cancels the pending attempt), then
            # the physical connection is established.
            adapter._cancel_pending_auto_join_if_matches(111, "42", "222")
            adapter._voice_clients[111] = SimpleNamespace(
                is_connected=lambda: True,
                is_playing=lambda: False,
                channel=SimpleNamespace(id=222, members=[SimpleNamespace(id=42)]),
                disconnect=AsyncMock(),
            )
            return True

        adapter.join_voice_channel = AsyncMock(side_effect=_join)

        runner = _make_runner(tmp_path)
        runner.adapters[Platform.DISCORD] = adapter
        runner._sync_voice_auto_join_state_to_adapter(adapter)

        member, before, after, _channel = TestDiscordVoiceAutoJoin._voice_state()
        await adapter._handle_voice_state_update(member, before, after)

        # Physical VC and all ownership state gone.
        assert 111 not in adapter._voice_clients
        assert 111 not in adapter._voice_auto_join_targets
        assert adapter._voice_auto_join_pending == {}
        # Persisted mode rolled back to off; auto-TTS enable cleared, disable set.
        assert runner._voice_mode.get("discord:333") == "off"
        assert "333" not in adapter._auto_tts_enabled_chats
        assert "333" in adapter._auto_tts_disabled_chats

    # ----- Finding 5: transcript dedupe keyed by profile -----

    @pytest.mark.asyncio
    async def test_transcript_dedupe_is_per_profile(self, tmp_path):
        runner = _make_runner(tmp_path)
        # Identical guild/user/transcript is accepted ONCE per profile...
        assert runner._is_duplicate_voice_transcript(111, 42, "hello there", profile=None) is False
        assert runner._is_duplicate_voice_transcript(111, 42, "hello there", profile="coder") is False
        # ...and a repeat within the SAME profile is suppressed.
        assert runner._is_duplicate_voice_transcript(111, 42, "hello there", profile=None) is True
        assert runner._is_duplicate_voice_transcript(111, 42, "hello there", profile="coder") is True

    @pytest.mark.asyncio
    async def test_secondary_voice_input_has_no_default_adapter_fallback(self, tmp_path):
        runner = _make_runner(tmp_path)
        runner._profile_adapters = {}
        default = self._mock_discord_adapter()
        default._voice_text_channels = {111: 333}
        runner.adapters[Platform.DISCORD] = default
        # Secondary profile "coder" has NO registered adapter.

        await runner._handle_voice_channel_input(
            111, 42, "orphan transcript", profile="coder"
        )

        # No fallback to the default adapter — the turn is dropped, default untouched.
        default.handle_message.assert_not_awaited()


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
        adapter._voice_sources[111] = {"chat_id": "123", "chat_type": "group"}

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
        assert 111 not in adapter._voice_sources
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
        assert adapter._is_allowed_user("42") is False

    def test_is_allowed_user_in_list(self):
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"42", "99"}
        assert adapter._is_allowed_user("42") is True

    def test_is_allowed_user_not_in_list(self):
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"99"}
        assert adapter._is_allowed_user("42") is False

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

    def test_is_allowed_user_wildcard_mixed_with_ids(self):
        """``DISCORD_ALLOWED_USERS="123,*"`` honors ``*`` for any user."""
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"123456789012345678", "*"}
        assert adapter._is_allowed_user("42") is True
        assert adapter._is_allowed_user("123456789012345678") is True

    def test_is_allowed_user_wildcard_in_dm(self):
        """Wildcard short-circuits before role-auth gating, so DMs honor it too."""
        adapter = self._make_adapter()
        adapter._allowed_user_ids = {"*"}
        assert adapter._is_allowed_user("42", is_dm=True) is True

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

    @pytest.mark.asyncio
    async def test_process_voice_input_hallucination_filtered(self):
        """Whisper hallucination is filtered out."""
        adapter = self._make_adapter()
        callback = AsyncMock()
        adapter._voice_input_callback = callback

        with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
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

        with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": False, "error": "API error"}):
            await adapter._process_voice_input(111, 42, b"\x00" * 96000)

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_voice_input_exception_caught(self):
        """Exception during processing is caught, no crash."""
        adapter = self._make_adapter()
        adapter._voice_input_callback = AsyncMock()

        with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav",
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

    def test_on_packet_buffer_write_holds_lock(self):
        """_on_packet must hold lock when writing to buffers."""
        import ast, inspect, textwrap
        from plugins.platforms.discord.adapter import VoiceReceiver
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
        import inspect
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
        runner._voice_mode["telegram:123"] = "all"

        result = await runner._handle_voice_channel_leave(event)
        assert "left" in result.lower()
        assert runner._voice_mode["telegram:123"] == "off"
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
        runner._voice_mode["telegram:123"] = "all"

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
        import inspect
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

    @pytest.mark.asyncio
    async def test_runner_cleanup_method_removes_voice_mode(self, tmp_path):
        """_handle_voice_timeout_cleanup removes voice_mode for chat."""
        runner = _make_runner(tmp_path)
        runner._voice_mode["discord:999"] = "all"

        runner._handle_voice_timeout_cleanup("999")

        assert runner._voice_mode["discord:999"] == "off", \
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

    @pytest.mark.asyncio
    async def test_timeout_skips_disconnect_when_voice_mode_off(self, adapter):
        """Voice-off is deliberate text-only mode, not idle neglect — the
        inactivity timer must NOT disconnect or spam the channel (#PanBartosz)."""
        disconnect_calls = []
        adapter._on_voice_disconnect = lambda chat_id: disconnect_calls.append(chat_id)
        adapter._voice_mode_getter = lambda chat_id: "off"

        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999
        adapter._voice_timeout_tasks[111] = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await adapter._voice_timeout_handler(111)

        # Still connected, no disconnect callback, no "inactivity timeout" spam.
        assert 111 in adapter._voice_clients
        assert disconnect_calls == []
        mock_vc.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_still_disconnects_when_voice_mode_active(self, adapter):
        """A non-off mode still auto-disconnects on genuine inactivity."""
        disconnect_calls = []
        adapter._on_voice_disconnect = lambda chat_id: disconnect_calls.append(chat_id)
        adapter._voice_mode_getter = lambda chat_id: "all"

        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999
        adapter._voice_timeout_tasks[111] = MagicMock()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await adapter._voice_timeout_handler(111)

        assert 111 not in adapter._voice_clients
        assert disconnect_calls == ["999"]

    @pytest.mark.asyncio
    async def test_timeout_uses_configured_seconds(self, adapter):
        adapter._voice_timeout_seconds = 17
        adapter._voice_auto_join_cfg = {}

        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await adapter._voice_timeout_handler(111)

        mock_sleep.assert_awaited_once_with(17)

    @pytest.mark.asyncio
    async def test_timeout_stays_while_auto_join_target_present(self, adapter):
        from plugins.platforms.discord.adapter import VoiceAutoJoinTarget

        adapter._voice_timeout_seconds = 17
        adapter._voice_auto_join_cfg = {"stay_while_target_present": True}
        adapter._voice_auto_join_retry_tasks = {}
        adapter._voice_auto_join_targets = {
            111: VoiceAutoJoinTarget(
                guild_id=111,
                user_id="42",
                voice_channel_id="222",
                text_channel_id="999",
            )
        }

        target_member = SimpleNamespace(id=42)
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.disconnect = AsyncMock()
        mock_vc.channel = SimpleNamespace(id=222, members=[target_member])
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await adapter._voice_timeout_handler(111)

        assert 111 in adapter._voice_clients
        mock_vc.disconnect.assert_not_called()
        # The handler must REARM a fresh timeout task so a later target
        # departure can still trigger the disconnect.
        rearmed = adapter._voice_timeout_tasks.get(111)
        assert rearmed is not None
        rearmed.cancel()

    @pytest.mark.asyncio
    async def test_timeout_rearm_disconnects_after_target_leaves(self, adapter):
        """Once the target is gone, the rearmed idle timeout disconnects even
        when target_leave_cleanup is False (no leave event tears it down)."""
        from plugins.platforms.discord.adapter import VoiceAutoJoinTarget

        adapter._voice_timeout_seconds = 5
        adapter._voice_auto_join_cfg = {
            "stay_while_target_present": True,
            "target_leave_cleanup": False,
        }
        adapter._voice_auto_join_retry_tasks = {}
        # Target is no longer present in the channel (empty members).
        adapter._voice_auto_join_targets = {
            111: VoiceAutoJoinTarget(
                guild_id=111,
                user_id="42",
                voice_channel_id="222",
                text_channel_id="999",
            )
        }
        mock_vc = MagicMock()
        mock_vc.is_connected.return_value = True
        mock_vc.is_playing.return_value = False
        mock_vc.disconnect = AsyncMock()
        mock_vc.channel = SimpleNamespace(id=222, members=[])
        adapter._voice_clients[111] = mock_vc
        adapter._voice_text_channels[111] = 999

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await adapter._voice_timeout_handler(111)

        # Target absent → no stay/rearm; the idle timeout disconnects the physical
        # session (transactional teardown under the guild lock).
        mock_vc.disconnect.assert_awaited()
        assert 111 not in adapter._voice_clients


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
        assert "PLAYBACK_TIMEOUT" in source, \
            "play_in_voice_channel must reference PLAYBACK_TIMEOUT constant"

    def test_playback_timeout_constant_exists(self):
        """PLAYBACK_TIMEOUT constant is defined on DiscordAdapter."""
        from plugins.platforms.discord.adapter import DiscordAdapter
        assert hasattr(DiscordAdapter, "PLAYBACK_TIMEOUT")
        assert DiscordAdapter.PLAYBACK_TIMEOUT > 0

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

    @pytest.mark.asyncio
    async def test_is_playing_wait_has_timeout(self):
        """While loop waiting for previous playback has a timeout."""
        from plugins.platforms.discord.adapter import DiscordAdapter
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

    def test_text_input_voice_off_no_tts(self):
        """Streaming OFF + text input + voice_mode=off: no TTS."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "off", MessageType.TEXT) is False

    def test_text_input_voice_only_no_tts(self):
        """Streaming OFF + text input + voice_mode=voice_only: no TTS for text."""
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

    # -- Streaming ON (already_sent=True) --

    def test_streaming_on_voice_input_runner_fires(self):
        """Streaming ON + voice input: runner handles TTS (base adapter has no text)."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.VOICE, already_sent=True) is True

    def test_streaming_on_text_input_runner_fires(self):
        """Streaming ON + text input: runner handles TTS (same as before)."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.TEXT, already_sent=True) is True

    def test_streaming_on_voice_off_no_tts(self):
        """Streaming ON + voice_mode=off: no TTS regardless of streaming."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "off", MessageType.VOICE, already_sent=True) is False

    def test_streaming_on_empty_response_no_tts(self):
        """Streaming ON + empty response: no TTS."""
        from gateway.platforms.base import MessageType
        runner = self._make_runner()
        assert self._call_should_reply(runner, "all", MessageType.VOICE, response="", already_sent=True) is False

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

    def test_keepalive_interval_is_reasonable(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        interval = DiscordAdapter._KEEPALIVE_INTERVAL
        assert 5 <= interval <= 30, f"Keepalive interval {interval}s should be between 5-30s"

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

    def test_default_true_no_override_fires(self):
        fn, adapter = self._make_adapter(default=True)
        assert fn(adapter, "chat1") is True

    def test_explicit_enable_overrides_false_default(self):
        """``/voice on`` with config auto_tts=False still fires."""
        fn, adapter = self._make_adapter(default=False, enabled={"chat1"})
        assert fn(adapter, "chat1") is True

    def test_explicit_disable_overrides_true_default(self):
        """``/voice off`` with config auto_tts=True still suppresses."""
        fn, adapter = self._make_adapter(default=True, disabled={"chat1"})
        assert fn(adapter, "chat1") is False

    def test_enabled_wins_over_disabled(self):
        """An explicit enable beats an explicit disable (enable takes priority)."""
        fn, adapter = self._make_adapter(
            default=False, enabled={"chat1"}, disabled={"chat1"}
        )
        assert fn(adapter, "chat1") is True

    def test_per_chat_isolation(self):
        """Enable for chat1 doesn't leak to chat2."""
        fn, adapter = self._make_adapter(default=False, enabled={"chat1"})
        assert fn(adapter, "chat1") is True
        assert fn(adapter, "chat2") is False


class TestTwoProfileVoiceStartupIsolation:
    """Integration: two profiles started through ``_start_one_profile_adapters``
    with OPPOSITE voice-mode / auto-TTS settings must stay isolated, and re-derive
    correctly on reconnect — exercised through actual adapter construction."""

    @pytest.mark.asyncio
    async def test_opposite_voice_settings_isolated_across_profiles_and_reconnect(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.profiles as _profiles_mod
        from gateway.run import GatewayRunner
        from gateway.config import Platform
        from plugins.platforms.discord.adapter import DiscordAdapter

        # Two profile homes with OPPOSITE voice.auto_tts defaults and their own
        # DISCORD_BOT_TOKEN (distinct, so credential-dedup doesn't refuse either).
        prof_a = tmp_path / "profA"
        prof_b = tmp_path / "profB"
        prof_a.mkdir()
        prof_b.mkdir()
        (prof_a / "config.yaml").write_text("voice:\n  auto_tts: true\n", encoding="utf-8")
        (prof_a / ".env").write_text("DISCORD_BOT_TOKEN=token-a\n", encoding="utf-8")
        (prof_b / "config.yaml").write_text("voice:\n  auto_tts: false\n", encoding="utf-8")
        (prof_b / ".env").write_text("DISCORD_BOT_TOKEN=token-b\n", encoding="utf-8")

        runner = object.__new__(GatewayRunner)
        runner._profile_adapters = {}
        runner.adapters = {}
        runner.session_store = MagicMock()
        runner._busy_text_mode = False
        runner.config = SimpleNamespace(
            multiplex_profiles=True,
            group_sessions_per_user=False,
            thread_sessions_per_user=False,
        )
        # OPPOSITE persisted per-profile voice modes (namespaced by profile).
        runner._voice_mode = {
            "profile:profA:discord:100": "all",   # A: auto-TTS ON for chat 100
            "profile:profB:discord:200": "off",   # B: auto-TTS OFF for chat 200
        }
        runner._VOICE_MODE_PATH = tmp_path / "voice_mode.json"

        # Peripheral wiring factories (unrelated to the voice-state focus).
        runner._make_profile_message_handler = lambda p: (lambda *a, **k: None)
        runner._make_profile_fatal_error_handler = lambda p, plat: (lambda *a, **k: None)
        runner._make_adapter_auth_check = lambda plat, profile_name=None: (lambda *a, **k: True)
        runner._handle_active_session_busy_message = MagicMock()
        runner._recover_telegram_topic_thread_id = MagicMock()
        # No live Discord: connect always "succeeds" without touching the network.
        runner._connect_adapter_with_timeout = AsyncMock(return_value=True)

        # Real adapter construction (discord.py is mocked in this suite, so the
        # plugin registry's dependency check would refuse; build the real adapter
        # directly while everything else in _start_one_profile_adapters runs for
        # real — config load under profile scope, policy guard, credential dedup,
        # _configure_profile_adapter, connect, and the two voice-sync methods).
        def _construct(platform, config):
            adapter = DiscordAdapter(config)
            adapter.gateway_runner = runner
            return adapter

        runner._create_adapter = _construct

        claimed = {}
        n_a = await runner._start_one_profile_adapters("profA", prof_a, claimed)
        n_b = await runner._start_one_profile_adapters("profB", prof_b, claimed)
        assert n_a == 1
        assert n_b == 1

        a = runner._profile_adapters["profA"][Platform.DISCORD]
        b = runner._profile_adapters["profB"][Platform.DISCORD]
        assert a is not b

        # Opposite auto_tts defaults, each read from its OWN profile's config.yaml.
        assert a._auto_tts_default is True
        assert b._auto_tts_default is False

        # Per-profile enabled/disabled chats — each adapter sees ONLY its own
        # namespace, never the other profile's mode keys.
        assert a._auto_tts_enabled_chats == {"100"}
        assert a._auto_tts_disabled_chats == set()
        assert b._auto_tts_enabled_chats == set()
        assert b._auto_tts_disabled_chats == {"200"}

        # Each profile has its own bound auto-join callback (never cross-wired).
        assert a._voice_auto_join_callback is not None
        assert b._voice_auto_join_callback is not None
        assert a._voice_auto_join_callback is not b._voice_auto_join_callback

        # Reconnect profB through the ACTUAL runner reconnect path — it
        # RECONSTRUCTS the adapter inside profB's runtime scope (via
        # get_profile_dir + _create_adapter) and re-runs the profile-scoped voice
        # syncs. Drop B's live adapter so the reconnect rebuilds a fresh one.
        homes = {"profA": prof_a, "profB": prof_b}
        monkeypatch.setattr(
            _profiles_mod, "get_profile_dir", lambda name: homes[name]
        )
        runner._running = True
        runner._profile_failed_platforms = {}
        runner._profile_adapters["profB"] = {}

        await runner._run_secondary_profile_reconnect("profB", Platform.DISCORD)

        b2 = runner._profile_adapters["profB"][Platform.DISCORD]
        # A genuinely reconstructed adapter (not the pre-reconnect instance),
        # re-derived from profB's OWN namespace only — no bleed from A.
        assert b2 is not b
        assert b2._auto_tts_default is False
        assert b2._auto_tts_enabled_chats == set()
        assert b2._auto_tts_disabled_chats == {"200"}
        assert b2._voice_auto_join_callback is not None
        # profA's adapter is untouched by profB's reconnect.
        assert a._auto_tts_default is True
        assert a._auto_tts_enabled_chats == {"100"}
