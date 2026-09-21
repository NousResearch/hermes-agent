"""Tests for the tts.provider=none short-circuit and fallback prevention.

When ``tts.provider in ('none', 'off', 'disabled', 'false')``:
1. ``text_to_speech_tool`` returns a clean error instead of synthesizing audio with edge-tts.
2. ``_select_builtin_engine`` refuses disabled providers instead of falling through to edge-tts.
3. ``BasePlatformAdapter._wants_auto_tts`` returns False without attempting auto-TTS.
4. ``GatewayVoiceMixin._should_send_voice_reply`` returns False without generating TTS.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
import json

import pytest

from gateway.config import Platform


class TestProviderNoneShortCircuit:
    """All TTS entry points must bail out cleanly when tts.provider is disabled."""

    @pytest.mark.parametrize("provider_value", ["none", "off", "disabled", "false",
                                                 "None", "OFF", "  none  "])
    def test_tts_tool_returns_error_for_disabled_provider(self, provider_value):
        """tools/tts_tool.py must refuse to synthesize when provider is disabled."""
        from tools.tts_tool import text_to_speech_tool
        result = text_to_speech_tool(text="Hello world", provider=provider_value)
        data = json.loads(result)
        assert data["success"] is False
        assert "disabled" in data.get("error", "").lower()

    @pytest.mark.parametrize("provider_value", ["none", "off", "disabled", "false"])
    def test_select_builtin_engine_refuses_disabled_provider(self, provider_value):
        """_select_builtin_engine must not fall through to edge-tts default."""
        from tools.tts_tool import _select_builtin_engine
        engine, err = _select_builtin_engine(provider_value)
        assert err is not None
        data = json.loads(err)
        assert data["success"] is False
        assert "disabled" in data.get("error", "").lower()

    @pytest.mark.parametrize("provider_value", ["none", "off", "disabled", "false"])
    def test_wants_auto_tts_returns_false_for_disabled_provider(self, provider_value, monkeypatch):
        """base.py _wants_auto_tts must return False when provider is disabled."""
        from gateway.platforms.base import BasePlatformAdapter
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.session import SessionSource
        import asyncio

        mock_config = {"tts": {"provider": provider_value}}
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: mock_config)

        adapter = MagicMock(spec=BasePlatformAdapter)
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
        adapter.platform = Platform.DISCORD

        event = MessageEvent(
            text="hello", message_type=MessageType.VOICE,
            source=SessionSource(platform=Platform.DISCORD, chat_id="test", chat_type="dm"),
        )
        interrupt = asyncio.Event()

        result = BasePlatformAdapter._wants_auto_tts(
            adapter, event, "session-key", interrupt, "reply text", [])
        assert result is False

    @pytest.mark.parametrize("provider_value", ["none", "off", "disabled", "false"])
    def test_should_send_voice_reply_returns_false_for_disabled_provider(
        self, provider_value, monkeypatch
    ):
        """run_voice.py _should_send_voice_reply must return False when provider is disabled."""
        mock_config = {"tts": {"provider": provider_value}}
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: mock_config)

        from gateway.run_voice import GatewayVoiceMixin
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.session import SessionSource

        mixin = object.__new__(GatewayVoiceMixin)
        mixin._voice_mode = {}
        mixin.config = SimpleNamespace(stt_echo_transcripts=True)

        event = MessageEvent(
            text="hello", message_type=MessageType.VOICE,
            source=SessionSource(platform=Platform.DISCORD, chat_id="test", chat_type="dm"),
        )

        result = mixin._should_send_voice_reply(event, "Hello world", [])
        assert result is False
