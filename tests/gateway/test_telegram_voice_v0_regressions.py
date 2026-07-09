import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter
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


@pytest.mark.asyncio
async def test_pending_voice_interrupt_reuses_transcript_and_echo():
    adapter = SimpleNamespace(send=AsyncMock())
    runner = _runner(adapter)
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/telegram-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello once", "provider": "mock"},
    ) as mock_transcribe:
        interrupt_text, interrupt_transcripts = await runner._transcribe_pending_audio_event_once(
            event,
            event.text,
        )
        await runner._echo_pending_stt_transcripts_once(
            event,
            adapter,
            source,
            interrupt_transcripts,
        )

        drain_text, drain_transcripts = await runner._transcribe_pending_audio_event_once(
            event,
            event.text,
        )
        await runner._echo_pending_stt_transcripts_once(
            event,
            adapter,
            source,
            drain_transcripts,
        )

    assert interrupt_text == '"hello once"'
    assert drain_text == interrupt_text
    assert drain_transcripts == interrupt_transcripts == ["hello once"]
    mock_transcribe.assert_called_once_with("/tmp/telegram-voice.ogg")
    adapter.send.assert_awaited_once_with(
        "12345",
        '🎙️ "hello once"',
        metadata=None,
    )


@pytest.mark.asyncio
async def test_busy_voice_interrupt_transcribes_before_pending_drain(monkeypatch):
    adapter = SimpleNamespace(send=AsyncMock(), _pending_messages={})
    runner = _runner(adapter)
    runner._is_user_authorized = lambda _source: True
    runner._draining = False
    runner._running_agents = {}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._busy_ack_ts = {}
    runner._queued_events = {}
    runner._agent_has_active_subagents = lambda _agent: False
    runner._session_has_compression_in_flight = lambda _session_key: False
    session_key = "telegram:dm:12345"
    agent = MagicMock()
    runner._running_agents[session_key] = agent
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/telegram-busy-voice.ogg"],
        media_types=["audio/ogg"],
    )
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    with (
        patch("tools.approval.has_blocking_approval", return_value=False),
        patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={"success": True, "transcript": "interrupt me", "provider": "mock"},
        ) as mock_transcribe,
    ):
        handled = await runner._handle_active_session_busy_message(event, session_key)
        drain_text, drain_transcripts = await runner._transcribe_pending_audio_event_once(
            adapter._pending_messages[session_key],
            event.text,
        )
        await runner._echo_pending_stt_transcripts_once(
            adapter._pending_messages[session_key],
            adapter,
            source,
            drain_transcripts,
        )

    assert handled is True
    agent.interrupt.assert_called_once_with('"interrupt me"')
    assert adapter._pending_messages[session_key] is event
    assert drain_text == '"interrupt me"'
    mock_transcribe.assert_called_once_with("/tmp/telegram-busy-voice.ogg")
    adapter.send.assert_awaited_once_with(
        "12345",
        '🎙️ "interrupt me"',
        metadata={},
    )


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
async def test_telegram_video_size_gate_rejects_oversized_media_before_download():
    adapter = object.__new__(TelegramAdapter)
    adapter._max_doc_bytes = 1024
    adapter._should_process_message = lambda _message: True
    adapter._build_message_event = lambda _message, _type, update_id=None: SimpleNamespace(
        text="caption",
        media_urls=[],
        media_types=[],
    )
    adapter._apply_telegram_group_observe_attribution = lambda event: event

    handled = []

    async def handle_message(event):
        handled.append(event)

    adapter.handle_message = handle_message

    class OversizedVideo:
        file_size = 2048

        async def get_file(self):  # pragma: no cover - failure path assertion
            pytest.fail("oversized videos must not be downloaded")

    msg = SimpleNamespace(
        caption=None,
        sticker=None,
        photo=None,
        voice=None,
        audio=None,
        video=OversizedVideo(),
        document=None,
        media_group_id=None,
    )
    update = SimpleNamespace(message=msg, update_id=1)

    await TelegramAdapter._handle_media_message(adapter, update, SimpleNamespace())

    assert len(handled) == 1
    assert handled[0].media_urls == []
    assert handled[0].media_types == []
    assert "video file" in handled[0].text
    assert "exceeds" in handled[0].text


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
