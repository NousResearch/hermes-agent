import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.telegram import TelegramAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")


def _runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        stt_enabled=True,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._consume_pending_native_image_paths = lambda _key: []
    runner._session_key_for_source = lambda _source: "telegram:dm:12345"
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._reply_anchor_for_event = lambda _event: None
    return runner


def test_telegram_audio_size_gate_rejects_oversized_media_before_download():
    adapter = object.__new__(TelegramAdapter)
    adapter._max_doc_bytes = 1024

    allowed, note = adapter._telegram_media_size_allowed(
        SimpleNamespace(file_size=2048),
        "voice message",
    )

    assert allowed is False
    assert "exceeds" in note
    assert "voice message" in note


@pytest.mark.asyncio
async def test_failed_voice_only_stt_sends_structured_notice_and_skips_agent(monkeypatch):
    sent = []
    adapter = SimpleNamespace(send=AsyncMock(side_effect=lambda chat_id, text, metadata=None: sent.append(text)))
    runner = _runner(adapter)

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda _path: {"success": False, "transcript": "", "error": "Request timeout: upstream"},
    )

    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )

    result = await runner._prepare_inbound_message_text(event=event, source=event.source, history=[])

    assert result is None
    assert sent
    assert "couldn't transcribe" in sent[0]
    assert "Request timeout" not in sent[0]


@pytest.mark.asyncio
async def test_failed_voice_with_caption_preserves_caption_without_prompt_error(monkeypatch):
    sent = []
    adapter = SimpleNamespace(send=AsyncMock(side_effect=lambda chat_id, text, metadata=None: sent.append(text)))
    runner = _runner(adapter)

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda _path: {"success": False, "transcript": "", "error": "API error: 429"},
    )

    event = MessageEvent(
        text="Use this caption instead",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )

    result = await runner._prepare_inbound_message_text(event=event, source=event.source, history=[])

    assert result == "Use this caption instead"
    assert sent
    assert "couldn't transcribe" in sent[0]
    assert "429" not in result


@pytest.mark.asyncio
async def test_voice_on_keeps_telegram_text_reply_first_regression():
    adapter = SimpleNamespace(
        _auto_tts_disabled_chats=set(),
        _auto_tts_enabled_chats=set(),
    )
    runner = _runner(adapter)
    runner._voice_mode = {}
    runner._voice_provider_mode = {}
    runner._save_voice_modes = lambda: None
    runner._save_voice_provider_modes = lambda: None

    event = SimpleNamespace(
        source=_source(),
        get_command_args=lambda: "on",
    )
    result = await GatewayRunner._handle_voice_command(runner, event)

    assert runner._voice_mode["telegram:12345"] == "voice_only"
    assert "12345" not in adapter._auto_tts_enabled_chats
    assert "12345" not in adapter._auto_tts_disabled_chats
    assert result


@pytest.mark.asyncio
async def test_voice_tts_is_explicit_audio_reply_opt_in():
    adapter = SimpleNamespace(
        _auto_tts_disabled_chats=set(),
        _auto_tts_enabled_chats=set(),
    )
    runner = _runner(adapter)
    runner._voice_mode = {}
    runner._voice_provider_mode = {}
    runner._save_voice_modes = lambda: None
    runner._save_voice_provider_modes = lambda: None

    event = SimpleNamespace(
        source=_source(),
        get_command_args=lambda: "tts",
    )
    result = await GatewayRunner._handle_voice_command(runner, event)

    assert runner._voice_mode["telegram:12345"] == "all"
    assert "12345" in adapter._auto_tts_enabled_chats
    assert result
