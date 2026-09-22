from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.run_voice import GatewayVoiceMixin
from gateway.slash_commands import GatewaySlashCommandsMixin


def test_clean_messaging_defaults_off():
    cfg = GatewayConfig.from_dict({})
    assert cfg.clean_messaging is False
    assert cfg.voice_include_text is True
    assert cfg.stt_echo_transcripts is True
    assert cfg.to_dict()["clean_messaging"] is False
    assert cfg.to_dict()["voice_include_text"] is True


def test_clean_messaging_master_switch():
    cfg = GatewayConfig.from_dict({"gateway": {"clean_messaging": True}})
    assert cfg.clean_messaging is True
    assert cfg.voice_include_text is False
    assert cfg.stt_echo_transcripts is False


def test_clean_messaging_granular_overrides():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "clean_messaging": True,
            "stt_echo_transcripts": True,
            "voice_include_text": True,
        }
    })
    assert cfg.clean_messaging is True
    assert cfg.stt_echo_transcripts is True
    assert cfg.voice_include_text is True


class DummyVoiceRunner(GatewayVoiceMixin):
    def __init__(self, config):
        self.config = config
        self._voice_mode = {}
        self._voice_text_mode = {}
        self.adapters = {}

    def _voice_key(self, platform, chat_id, profile=None):
        return f"{platform.value}:{chat_id}"

    def _adapter_profile_for_source(self, source):
        return "default"


def test_should_echo_stt_transcripts_clean_messaging():
    runner = DummyVoiceRunner(GatewayConfig.from_dict({"gateway": {"clean_messaging": True}}))
    assert runner._should_echo_stt_transcripts() is False

    runner_normal = DummyVoiceRunner(GatewayConfig.from_dict({}))
    assert runner_normal._should_echo_stt_transcripts() is True


def test_should_send_voice_text_and_per_chat_override(tmp_path):
    runner = DummyVoiceRunner(GatewayConfig.from_dict({"gateway": {"clean_messaging": True}}))
    runner._VOICE_TEXT_MODE_PATH = tmp_path / "voice_text_mode.json"

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345")
    event = MessageEvent(text="hi", message_type=MessageType.VOICE, source=source)

    # Clean messaging defaults to audio-only (False)
    assert runner._should_send_voice_text(event) is False

    # Override for this chat to include text
    key = runner._voice_key_for_source(source)
    runner._set_voice_text_mode(key, True)
    assert runner._should_send_voice_text(event) is True

    # Override back to audio-only
    runner._set_voice_text_mode(key, False)
    assert runner._should_send_voice_text(event) is False


class DummySlashRunner(GatewaySlashCommandsMixin, GatewayVoiceMixin):
    def __init__(self, config):
        self.config = config
        self._voice_mode = {}
        self._voice_text_mode = {}
        self.adapters = {}

    def _delivery_adapter_for(self, source):
        adapter = MagicMock()
        adapter.get_voice_channel_info.return_value = None
        return adapter

    def _voice_key_for_source(self, source):
        return f"{source.platform.value}:{source.chat_id}"

    def _voice_key(self, platform, chat_id, profile=None):
        return f"{platform.value}:{chat_id}"

    def _adapter_profile_for_source(self, source):
        return "default"

    def _get_guild_id(self, event):
        return None


def test_voice_slash_command_text_toggle(tmp_path):
    async def run_test():
        runner = DummySlashRunner(GatewayConfig.from_dict({}))
        runner._VOICE_MODE_PATH = tmp_path / "voice_mode.json"
        runner._VOICE_TEXT_MODE_PATH = tmp_path / "voice_text_mode.json"

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345")
        
        event_off = MessageEvent(text="/voice text off", message_type=MessageType.COMMAND, source=source)
        res_off = await runner._handle_voice_command(event_off)
        assert "audio-only" in res_off.lower()
        assert runner._should_send_voice_text(event_off) is False

        event_status = MessageEvent(text="/voice status", message_type=MessageType.COMMAND, source=source)
        res_status = await runner._handle_voice_command(event_status)
        assert "audio only" in res_status.lower()

        event_on = MessageEvent(text="/voice text on", message_type=MessageType.COMMAND, source=source)
        res_on = await runner._handle_voice_command(event_on)
        assert "accompanying text" in res_on.lower()
        assert runner._should_send_voice_text(event_on) is True

    import asyncio
    asyncio.run(run_test())


class DummyTurnRunner(GatewayVoiceMixin):
    def __init__(self, config):
        self.config = config
        self._voice_mode = {}
        self._voice_text_mode = {}
        self.adapters = {}

    def _delivery_adapter_for(self, source):
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter.send_voice = AsyncMock()
        adapter._streaming_tts_turn_completed.return_value = False
        return adapter

    def _voice_key_for_source(self, source):
        return f"{source.platform.value}:{source.chat_id}"

    def _voice_key(self, platform, chat_id, profile=None):
        return f"{platform.value}:{chat_id}"

    def _should_send_voice_reply(self, event, response, agent_messages, already_sent=False):
        return True

    def _event_thread_metadata(self, event, source):
        return {}

    async def _deliver_media_from_response(self, response, event, adapter):
        pass


def test_deliver_turn_response_voice_only_suppresses_duplicate_text():
    from gateway.run_turn import GatewayTurnMixin

    class FullTurnRunner(GatewayTurnMixin, DummyTurnRunner):
        pass

    async def run_test():
        runner = FullTurnRunner(GatewayConfig.from_dict({"gateway": {"clean_messaging": True}}))
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345")
        event = MessageEvent(text="hi", message_type=MessageType.VOICE, source=source)
        session_entry = SimpleNamespace(session_id="test_session")

        # Mock voice reply succeeding
        runner._send_voice_reply = AsyncMock(return_value=True)

        result = await runner._hmwa_deliver_turn_response(
            event=event,
            source=source,
            session_entry=session_entry,
            session_key="test_key",
            run_generation=1,
            agent_result={"already_sent": False},
            agent_messages=[],
            response="This is the response text",
            _footer_line=None,
            _intentional_silence=False,
        )
        # Duplicate text must be suppressed (None returned)
        assert result is None

    import asyncio
    asyncio.run(run_test())


def test_deliver_turn_response_voice_failure_falls_back_to_text():
    from gateway.run_turn import GatewayTurnMixin

    class FullTurnRunner(GatewayTurnMixin, DummyTurnRunner):
        pass

    async def run_test():
        runner = FullTurnRunner(GatewayConfig.from_dict({"gateway": {"clean_messaging": True}}))
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345")
        event = MessageEvent(text="hi", message_type=MessageType.VOICE, source=source)
        session_entry = SimpleNamespace(session_id="test_session")

        # Mock voice reply failing (e.g. TTS API error)
        runner._send_voice_reply = AsyncMock(return_value=False)

        result = await runner._hmwa_deliver_turn_response(
            event=event,
            source=source,
            session_entry=session_entry,
            session_key="test_key",
            run_generation=1,
            agent_result={"already_sent": False},
            agent_messages=[],
            response="This is the response text fallback",
            _footer_line=None,
            _intentional_silence=False,
        )
        # On TTS failure, text must be returned as fallback so user gets an answer!
        assert result == "This is the response text fallback"

    import asyncio
    asyncio.run(run_test())
