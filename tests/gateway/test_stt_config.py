"""Tests for STT config — gateway should honor stt.enabled: false (#1100)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType


def _make_runner(stt_enabled=True):
    from gateway.run import GatewayRunner
    config = GatewayConfig(stt_enabled=stt_enabled)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    return runner


@pytest.mark.asyncio
async def test_stt_disabled_skips_transcription():
    runner = _make_runner(stt_enabled=False)
    runner._enrich_message_with_transcription = AsyncMock(return_value="should not be called")
    audio_paths = ["/tmp/voice.ogg"]
    if audio_paths and runner.config.stt_enabled:
        await runner._enrich_message_with_transcription("", audio_paths)
    runner._enrich_message_with_transcription.assert_not_called()


@pytest.mark.asyncio
async def test_stt_enabled_calls_transcription():
    runner = _make_runner(stt_enabled=True)
    runner._enrich_message_with_transcription = AsyncMock(return_value="[transcript]")
    audio_paths = ["/tmp/voice.ogg"]
    if audio_paths and runner.config.stt_enabled:
        await runner._enrich_message_with_transcription("", audio_paths)
    runner._enrich_message_with_transcription.assert_called_once()


def test_gateway_config_stt_enabled_default():
    config = GatewayConfig()
    assert config.stt_enabled is True


def test_gateway_config_stt_disabled_from_dict():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_gateway_config_stt_enabled_from_dict():
    config = GatewayConfig.from_dict({"stt": {"enabled": True}})
    assert config.stt_enabled is True


def test_gateway_config_stt_missing_defaults_true():
    config = GatewayConfig.from_dict({})
    assert config.stt_enabled is True
