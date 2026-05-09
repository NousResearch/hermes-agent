from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, SessionStore


def _make_runner(*, stt_config=None, adapter=None, tmp_path=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    if stt_config is not None:
        runner.config.stt = stt_config
    runner.adapters = {Platform.TELEGRAM: adapter or SimpleNamespace(send=AsyncMock())}
    runner._pending_native_image_paths_by_session = {}
    runner.session_store = SessionStore(sessions_dir=tmp_path, config=runner.config)
    runner._decide_image_input_mode = lambda: "text"
    runner._has_setup_skill = lambda: False
    return runner


def _make_voice_event():
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
            user_id="456",
            user_name="tester",
        ),
        message_id="99",
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )


def test_gateway_config_preserves_stt_mapping_for_transcript_echo():
    cfg = GatewayConfig.from_dict(
        {
            "platforms": {"telegram": {"enabled": True}},
            "stt": {"enabled": True, "echo_transcript": True, "echo_mode": "separate"},
        }
    )

    assert cfg.stt == {"enabled": True, "echo_transcript": True, "echo_mode": "separate"}


@pytest.mark.asyncio
async def test_stt_echo_prefix_prepends_user_visible_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "please check the backups"},
    )
    runner = _make_runner(stt_config={"echo_transcript": True, "echo_mode": "prefix"}, tmp_path=tmp_path)

    text = await runner._prepare_inbound_message_text(
        event=_make_voice_event(),
        source=_make_voice_event().source,
        history=[],
    )

    assert text.startswith('Heard: "please check the backups"\n\n')
    assert '[The user sent a voice message~ Here\'s what they said: "please check the backups"]' in text


@pytest.mark.asyncio
async def test_stt_echo_separate_sends_transcript_before_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "restart the gateway"},
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner = _make_runner(stt_config={"echo_transcript": True, "echo_mode": "separate"}, adapter=adapter, tmp_path=tmp_path)
    event = _make_voice_event()

    text = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    adapter.send.assert_awaited_once_with(
        "123",
        'Heard: "restart the gateway"',
        metadata=None,
    )
    assert text == '[The user sent a voice message~ Here\'s what they said: "restart the gateway"]'


@pytest.mark.asyncio
async def test_stt_echo_separate_uses_explicit_echo_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "topic voice note"},
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner = _make_runner(stt_config={"echo_transcript": True, "echo_mode": "separate"}, adapter=adapter, tmp_path=tmp_path)
    event = _make_voice_event()

    await runner._enrich_message_with_transcription(
        "",
        event.media_urls,
        source=event.source,
        echo_metadata={"thread_id": "63014", "telegram_reply_to_message_id": 999},
    )

    adapter.send.assert_awaited_once_with(
        "123",
        'Heard: "topic voice note"',
        metadata={"thread_id": "63014", "telegram_reply_to_message_id": 999},
    )


@pytest.mark.asyncio
async def test_stt_echo_separate_uses_thread_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "threaded voice note"},
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner = _make_runner(stt_config={"echo_transcript": True, "echo_mode": "separate"}, adapter=adapter, tmp_path=tmp_path)
    event = _make_voice_event()
    event.source.thread_id = "61892"

    await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    adapter.send.assert_awaited_once_with(
        "123",
        'Heard: "threaded voice note"',
        metadata={
            "thread_id": "61892",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "99",
        },
    )
