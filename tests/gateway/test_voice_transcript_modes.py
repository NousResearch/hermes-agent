"""Tests for voice-channel transcript routing and transcribe-only mode.

Covers:
- ``GatewayRunner._voice_transcript_settings`` resolution (config defaults,
  invalid-value fallback, per-chat slash prefs overriding config)
- ``/voice transcribe on|off`` and ``/voice transcripts channel|file|both``
  slash handling and persistence
- ``_handle_voice_channel_input`` routing: channel/file/both destinations and
  the transcribe-only agent gate
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Reuse the discord mock installer from the voice command tests so this file
# is runnable standalone and under the hermetic CI wrapper alike.
from tests.gateway.test_voice_command import _ensure_discord_mock

_ensure_discord_mock()

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


DISCORD_KEY = f"{Platform.DISCORD.value}:42"


def _make_event(text: str, chat_id: str = "42") -> MessageEvent:
    source = SessionSource(
        chat_id=chat_id,
        user_id="user1",
        platform=Platform.DISCORD,
    )
    source.thread_id = None
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    event.message_id = "msg42"
    return event


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._voice_transcript_prefs = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._VOICE_TRANSCRIPT_PREFS_PATH = tmp_path / "gateway_voice_transcript_prefs.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    return runner


def _make_discord_adapter(text_ch_id: int = 42):
    adapter = MagicMock()
    adapter._voice_text_channels = {1: text_ch_id}
    adapter._voice_sources = {}
    channel = MagicMock()
    channel.send = AsyncMock()
    adapter._client.get_channel.return_value = channel
    adapter._client.get_user.return_value = None
    adapter._resolve_channel_prompt = lambda chat_id: None
    adapter.handle_message = AsyncMock()
    return adapter, channel


# =====================================================================
# _voice_transcript_settings resolution
# =====================================================================

class TestVoiceTranscriptSettings:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    def test_defaults(self, runner):
        with patch("hermes_cli.config.load_config", return_value={}):
            destination, transcript_dir, to_agent = runner._voice_transcript_settings(
                DISCORD_KEY
            )
        assert destination == "channel"
        assert transcript_dir.endswith("transcripts")
        assert to_agent is True

    def test_config_values_respected(self, runner, tmp_path):
        cfg = {
            "discord": {
                "voice_transcript_destination": "file",
                "voice_transcript_dir": str(tmp_path / "vc-logs"),
                "voice_transcript_agent_turns": False,
            }
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            destination, transcript_dir, to_agent = runner._voice_transcript_settings(
                DISCORD_KEY
            )
        assert destination == "file"
        assert transcript_dir == str(tmp_path / "vc-logs")
        assert to_agent is False

    def test_invalid_destination_falls_back_to_channel(self, runner):
        cfg = {"discord": {"voice_transcript_destination": "carrier-pigeon"}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            destination, _, _ = runner._voice_transcript_settings(DISCORD_KEY)
        assert destination == "channel"

    def test_chat_prefs_override_config(self, runner):
        cfg = {
            "discord": {
                "voice_transcript_destination": "channel",
                "voice_transcript_agent_turns": True,
            }
        }
        runner._voice_transcript_prefs[DISCORD_KEY] = {
            "destination": "file",
            "to_agent": False,
        }
        with patch("hermes_cli.config.load_config", return_value=cfg):
            destination, _, to_agent = runner._voice_transcript_settings(DISCORD_KEY)
        assert destination == "file"
        assert to_agent is False

    def test_prefs_for_other_chat_do_not_leak(self, runner):
        runner._voice_transcript_prefs["discord:99"] = {"to_agent": False}
        with patch("hermes_cli.config.load_config", return_value={}):
            _, _, to_agent = runner._voice_transcript_settings(DISCORD_KEY)
        assert to_agent is True


# =====================================================================
# /voice transcribe and /voice transcripts slash handling
# =====================================================================

class TestVoiceTranscribeCommand:

    @pytest.fixture
    def runner(self, tmp_path):
        return _make_runner(tmp_path)

    @pytest.mark.asyncio
    async def test_transcribe_on(self, runner):
        event = _make_event("/voice transcribe on")
        result = await runner._handle_voice_command(event)
        assert runner._voice_transcript_prefs[DISCORD_KEY]["to_agent"] is False
        assert "transcribe" in result.lower()

    @pytest.mark.asyncio
    async def test_transcribe_off(self, runner):
        runner._voice_transcript_prefs[DISCORD_KEY] = {"to_agent": False}
        event = _make_event("/voice transcribe off")
        await runner._handle_voice_command(event)
        assert runner._voice_transcript_prefs[DISCORD_KEY]["to_agent"] is True

    @pytest.mark.asyncio
    async def test_transcribe_persists(self, runner):
        event = _make_event("/voice transcribe on")
        await runner._handle_voice_command(event)
        assert runner._VOICE_TRANSCRIPT_PREFS_PATH.exists()
        data = json.loads(runner._VOICE_TRANSCRIPT_PREFS_PATH.read_text())
        assert data[DISCORD_KEY]["to_agent"] is False

    @pytest.mark.asyncio
    async def test_transcribe_bare_reports_state(self, runner):
        with patch("hermes_cli.config.load_config", return_value={}):
            result = await runner._handle_voice_command(
                _make_event("/voice transcribe")
            )
        assert "off" in result.lower()

    @pytest.mark.asyncio
    async def test_transcripts_file(self, runner):
        with patch("hermes_cli.config.load_config", return_value={}):
            result = await runner._handle_voice_command(
                _make_event("/voice transcripts file")
            )
        assert runner._voice_transcript_prefs[DISCORD_KEY]["destination"] == "file"
        assert "transcripts" in result.lower()

    @pytest.mark.asyncio
    async def test_transcripts_invalid_value_reports_status(self, runner):
        with patch("hermes_cli.config.load_config", return_value={}):
            result = await runner._handle_voice_command(
                _make_event("/voice transcripts carrier-pigeon")
            )
        assert DISCORD_KEY not in runner._voice_transcript_prefs or (
            "destination" not in runner._voice_transcript_prefs[DISCORD_KEY]
        )
        assert "channel" in result.lower()

    @pytest.mark.asyncio
    async def test_prefs_round_trip_through_loader(self, runner, tmp_path):
        await runner._handle_voice_command(_make_event("/voice transcribe on"))
        await runner._handle_voice_command(_make_event("/voice transcripts both"))
        fresh = _make_runner(tmp_path)
        loaded = fresh._load_voice_transcript_prefs()
        assert loaded[DISCORD_KEY] == {"to_agent": False, "destination": "both"}


# =====================================================================
# _handle_voice_channel_input routing
# =====================================================================

class TestVoiceChannelInputRouting:

    @pytest.fixture
    def runner(self, tmp_path):
        runner = _make_runner(tmp_path)
        self_adapter, self_channel = _make_discord_adapter()
        runner.adapters = {Platform.DISCORD: self_adapter}
        self.adapter = self_adapter
        self.channel = self_channel
        return runner

    def _cfg(self, tmp_path, destination, to_agent=True):
        return {
            "discord": {
                "voice_transcript_destination": destination,
                "voice_transcript_dir": str(tmp_path / "vc-logs"),
                "voice_transcript_agent_turns": to_agent,
            }
        }

    @pytest.mark.asyncio
    async def test_channel_destination_posts_no_file(self, runner, tmp_path):
        cfg = self._cfg(tmp_path, "channel")
        with patch("hermes_cli.config.load_config", return_value=cfg):
            await runner._handle_voice_channel_input(1, 777, "hello there")
        self.channel.send.assert_awaited_once()
        assert "hello there" in self.channel.send.await_args.args[0]
        assert not (tmp_path / "vc-logs").exists()
        self.adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_file_destination_writes_no_post(self, runner, tmp_path):
        cfg = self._cfg(tmp_path, "file")
        with patch("hermes_cli.config.load_config", return_value=cfg):
            await runner._handle_voice_channel_input(1, 777, "for the record")
        self.channel.send.assert_not_awaited()
        logs = list((tmp_path / "vc-logs").glob("vc-1-*.log"))
        assert len(logs) == 1
        content = logs[0].read_text()
        assert "for the record" in content
        assert "777" in content

    @pytest.mark.asyncio
    async def test_both_destination_posts_and_writes(self, runner, tmp_path):
        cfg = self._cfg(tmp_path, "both")
        with patch("hermes_cli.config.load_config", return_value=cfg):
            await runner._handle_voice_channel_input(1, 777, "belt and suspenders")
        self.channel.send.assert_awaited_once()
        logs = list((tmp_path / "vc-logs").glob("vc-1-*.log"))
        assert len(logs) == 1

    @pytest.mark.asyncio
    async def test_transcribe_only_skips_agent(self, runner, tmp_path):
        cfg = self._cfg(tmp_path, "channel", to_agent=False)
        with patch("hermes_cli.config.load_config", return_value=cfg):
            await runner._handle_voice_channel_input(1, 777, "just listening")
        self.channel.send.assert_awaited_once()
        self.adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slash_pref_gates_agent_over_config(self, runner, tmp_path):
        cfg = self._cfg(tmp_path, "channel", to_agent=True)
        runner._voice_transcript_prefs[DISCORD_KEY] = {"to_agent": False}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            await runner._handle_voice_channel_input(1, 777, "pref wins")
        self.adapter.handle_message.assert_not_awaited()
