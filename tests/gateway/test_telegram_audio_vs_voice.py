"""
Tests for #24870 — Telegram: audio file attachments must NOT be routed to STT.

Telegram distinguishes three kinds of audio payloads:
  - message.voice  → Opus/OGG voice message  → STT pipeline
  - message.audio  → audio file attachment   → file path note, NOT STT
  - message.document (audio mime) → generic file route

These tests confirm that:
  1. MessageType.VOICE events still flow through the STT pipeline.
  2. MessageType.AUDIO events bypass STT and get a file-path context note instead.
  3. Mixed media lists (voice + audio) split correctly.
"""

from unittest.mock import AsyncMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import _propagate_pending_stt_reply_anchor
from gateway.session import SessionSource


def _make_runner(stt_enabled: bool = True) -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=stt_enabled)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _voice_event(path: str = "/tmp/voice.ogg") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["audio/ogg"],
    )


def _audio_event(path: str = "/tmp/song.mp3") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.AUDIO,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["audio/mpeg"],
    )


# ---------------------------------------------------------------------------
# 1. VOICE still goes through STT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_message_still_transcribed():
    """MessageType.VOICE must still be sent through _enrich_message_with_transcription."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _voice_event("/tmp/voice.ogg")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world", "provider": "whisper"},
    ) as mock_transcribe:
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    mock_transcribe.assert_called_once_with("/tmp/voice.ogg", None, "gateway")
    # The transcript passes through as a plain quoted line — no "voice message"
    # meta-commentary in the LLM-visible prompt.
    assert "hello world" in result


@pytest.mark.asyncio
async def test_telegram_reply_can_anchor_to_transcript_echo():
    runner = _make_runner(stt_enabled=True)
    transcript_adapter = AsyncMock()
    transcript_adapter.config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"reply_to_transcript": True},
    )
    transcript_adapter.send.return_value = SendResult(
        success=True,
        message_id="transcript-echo-7",
    )
    runner.adapters = {Platform.TELEGRAM: transcript_adapter}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _voice_event("/tmp/voice.ogg")
    event.message_id = "voice-note-6"

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world", "provider": "whisper"},
    ):
        await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    transcript_adapter.send.assert_awaited_once()
    assert runner._reply_anchor_for_event(event) == "transcript-echo-7"


def test_pending_voice_echo_metadata_keeps_telegram_dm_topic_lane():
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1",
        chat_type="dm",
        thread_id="42",
    )
    event = _voice_event()
    event.source = source
    event.message_id = "voice-note-6"

    metadata = runner._pending_voice_echo_metadata(event, source)

    assert metadata == {
        "thread_id": "42",
        "telegram_dm_topic_reply_fallback": True,
        "direct_messages_topic_id": "42",
        "telegram_reply_to_message_id": "voice-note-6",
    }


def test_latest_queued_transcript_anchor_propagates_to_outer_delivery():
    pending_event = _voice_event()
    setattr(pending_event, "_gateway_stt_reply_anchor", "transcript-echo-9")

    result = _propagate_pending_stt_reply_anchor(
        pending_event,
        {"final_response": "queued answer"},
    )

    assert result["_gateway_stt_reply_anchor"] == "transcript-echo-9"


def test_deeper_queued_transcript_anchor_wins_over_earlier_followup():
    pending_event = _voice_event()
    setattr(pending_event, "_gateway_stt_reply_anchor", "transcript-echo-8")

    result = _propagate_pending_stt_reply_anchor(
        pending_event,
        {
            "final_response": "latest queued answer",
            "_gateway_stt_reply_anchor": "transcript-echo-9",
        },
    )

    assert result["_gateway_stt_reply_anchor"] == "transcript-echo-9"


def test_one_level_queued_text_uses_ordinary_message_id():
    pending_event = MessageEvent(
        text="latest text",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="1",
            chat_type="dm",
        ),
        message_id="text-message-9",
    )

    result = _propagate_pending_stt_reply_anchor(
        pending_event,
        {"final_response": "queued answer"},
    )

    assert result["_gateway_stt_reply_anchor"] == "text-message-9"


def test_queued_turn_preserves_effective_none_reply_anchor():
    """Platform-specific top-level/topic routing must survive queued recursion."""
    pending_event = MessageEvent(
        text="latest text",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="1",
            chat_type="group",
            thread_id="42",
        ),
        message_id="topic-message-9",
    )

    result = _propagate_pending_stt_reply_anchor(
        pending_event,
        {"final_response": "queued answer"},
        effective_anchor=None,
    )

    assert "_gateway_stt_reply_anchor" in result
    assert result["_gateway_stt_reply_anchor"] is None


@pytest.mark.parametrize(
    "echo_result",
    [
        SendResult(success=False, error="echo failed"),
        SendResult(success=True, message_id=None),
    ],
    ids=["failed", "missing-message-id"],
)
@pytest.mark.asyncio
async def test_one_level_unanchored_voice_echo_uses_ordinary_message_id(echo_result):
    runner = _make_runner(stt_enabled=True)
    transcript_adapter = AsyncMock()
    transcript_adapter.config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"reply_to_transcript": True},
    )
    transcript_adapter.send.return_value = echo_result
    runner.adapters = {Platform.TELEGRAM: transcript_adapter}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1",
        chat_type="dm",
    )
    pending_event = _voice_event()
    pending_event.message_id = "voice-message-9"

    await runner._echo_pending_stt_transcripts_once(
        pending_event,
        transcript_adapter,
        source,
        ["latest voice"],
    )
    result = _propagate_pending_stt_reply_anchor(
        pending_event,
        {"final_response": "queued answer"},
    )

    assert not hasattr(pending_event, "_gateway_stt_reply_anchor")
    assert result["_gateway_stt_reply_anchor"] == "voice-message-9"


def test_multilevel_deeper_failed_echo_discards_earlier_successful_echo():
    earlier_event = _voice_event()
    earlier_event.message_id = "voice-message-8"
    setattr(earlier_event, "_gateway_stt_reply_anchor", "transcript-echo-8")
    deepest_event = _voice_event()
    deepest_event.message_id = "voice-message-9"

    deepest_result = _propagate_pending_stt_reply_anchor(
        deepest_event,
        {"final_response": "deepest queued answer"},
    )
    result = _propagate_pending_stt_reply_anchor(earlier_event, deepest_result)

    assert result["_gateway_stt_reply_anchor"] == "voice-message-9"


def test_multilevel_deepest_text_discards_earlier_successful_echo():
    earlier_event = _voice_event()
    earlier_event.message_id = "voice-message-8"
    setattr(earlier_event, "_gateway_stt_reply_anchor", "transcript-echo-8")
    deepest_event = MessageEvent(
        text="deepest text",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="1",
            chat_type="dm",
        ),
        message_id="text-message-9",
    )

    deepest_result = _propagate_pending_stt_reply_anchor(
        deepest_event,
        {"final_response": "deepest queued answer"},
    )
    result = _propagate_pending_stt_reply_anchor(earlier_event, deepest_result)

    assert result["_gateway_stt_reply_anchor"] == "text-message-9"


def test_consumed_queued_anchor_does_not_leak_in_agent_result():
    event = _voice_event()
    result = {
        "final_response": "deepest queued answer",
        "_gateway_stt_reply_anchor": "text-message-9",
    }

    consume_anchor = getattr(
        gateway_run,
        "_consume_pending_stt_reply_anchor",
        None,
    )
    assert consume_anchor is not None
    consume_anchor(event, result)

    assert getattr(event, "_gateway_stt_reply_anchor") == "text-message-9"
    assert result == {"final_response": "deepest queued answer"}


def test_consumed_anchor_tombstone_clears_stale_outer_anchor():
    event = _voice_event()
    setattr(event, "_gateway_stt_reply_anchor", "transcript-echo-8")
    result = {
        "final_response": "deepest queued answer",
        "_gateway_stt_reply_anchor": None,
    }

    consume_anchor = getattr(
        gateway_run,
        "_consume_pending_stt_reply_anchor",
        None,
    )
    assert consume_anchor is not None
    consume_anchor(event, result)

    assert getattr(event, "_gateway_stt_reply_anchor") is None
    assert result == {"final_response": "deepest queued answer"}


# ---------------------------------------------------------------------------
# 2. AUDIO file attachment bypasses STT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_attachment_context_note_format():
    """Context note for audio file attachments should include the file path and guidance."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _audio_event("/tmp/cache_12345_my_song.mp3")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("must not be called"),
    ):
        with patch(
            "tools.credential_files.to_agent_visible_cache_path",
            side_effect=lambda p: p,
        ):
            result = await runner._prepare_inbound_message_text(
                event=event,
                source=source,
                history=[],
            )

    assert "my_song.mp3" in result
    assert "audio file attachment" in result.lower()
    # Should NOT contain the voice-message transcription wrapper text
    assert "voice message" not in result.lower()
    # Guides the agent to transcribe/process the file itself rather than
    # punting back to the user (same bug class as the PDF/DOCX note).
    assert "transcri" in result.lower()
    assert "ask the user what they'd like" not in result.lower()


# ---------------------------------------------------------------------------
# 3. STT disabled still results in no transcription for audio file attachments
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Telegram gateway: msg.audio → MessageType.AUDIO (not VOICE)
# ---------------------------------------------------------------------------

