"""STT origin survives gateway preparation without changing model-facing text."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.turn_context import _stage_turn_user_message
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True, stt_echo_transcripts=False)
    runner.adapters = {}
    runner._model = "synthetic-model"
    runner._base_url = ""
    return runner


def _event(path, message_id):
    return MessageEvent(
        text="", message_type=MessageType.VOICE,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic-room", chat_type="dm"),
        message_id=message_id, media_urls=[str(path)], media_types=["audio/ogg"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("outcome", ["configured", "fallback", "empty", "failed"])
async def test_prepared_voice_keeps_origin_and_text_contract(tmp_path, pending, outcome):
    runner = _runner()
    event = _event(tmp_path / "clip.ogg", "voice-17")
    transcript = "The observatory booking is Friday at seven."
    success = {"success": True, "transcript": transcript}
    configured = success if outcome == "configured" else (
        {"success": True, "transcript": " "} if outcome == "empty" else {"success": False}
    )
    fallback = success if outcome == "fallback" else {"success": False}
    with patch("tools.transcription_tools.transcribe_audio", return_value=configured) as stt, patch(
        "tools.transcription_tools.transcribe_audio_local_fallback", return_value=fallback,
    ):
        if pending:
            await runner._transcribe_pending_audio_event_once(event)
        text = await runner._prepare_inbound_message_text(event=event, source=event.source, history=[])
        stt.assert_called_once()
    record, = event.metadata["audio_transcriptions"]
    assert record["source_path"] == event.media_urls[0]
    assert record["source_message_id"] == event.message_id
    assert record["confidence"] is None
    assert record["kind"] == "audio_transcript"
    if outcome in {"configured", "fallback"}:
        assert text == f'"{transcript}"'
        assert record["transcript"] == transcript
        assert record["status"] == "transcribed"
        assert record["method"] == ("local_fallback" if outcome == "fallback" else "configured")
    else:
        assert "transcript" not in record
        assert record["status"] == ("empty" if outcome == "empty" else "failed")
        assert transcript not in text

    from gateway.transcription_metadata import user_display_metadata

    metadata = user_display_metadata(event, persistence_owner="test-owner")
    row, _ = _stage_turn_user_message(SimpleNamespace(), text, text, 123.0, event.message_id, None, metadata)
    prepared = SimpleNamespace(persist_user_message=text, message_text=text, persist_user_timestamp=123.0,
                               persist_user_display_kind=None, persistence_owner="test-owner")
    failure_row = runner._hmwa_user_transcript_entry(event, prepared, 124.0)
    assert row["content"] == failure_row["content"] == text
    assert row["display_metadata"] == failure_row["display_metadata"] == metadata
    event.metadata["audio_transcriptions"].clear()
    assert row["display_metadata"]["audio_transcriptions"] == [record]


@pytest.mark.asyncio
async def test_merged_pending_clips_keep_their_own_origins(tmp_path):
    runner = _runner()
    first = _event(tmp_path / "first.ogg", "voice-1")
    second = _event(tmp_path / "second.ogg", "voice-2")
    queue = {"session-a": first}
    with patch("tools.transcription_tools.transcribe_audio", side_effect=lambda path, *_: {
        "success": True, "transcript": "Morning" if path == first.media_urls[0] else "Evening",
    }) as stt:
        await runner._transcribe_pending_audio_event_once(first)
        merge_pending_message_event(queue, "session-a", second)
        assert "audio_transcriptions" not in first.metadata
        text, _ = await runner._transcribe_pending_audio_event_once(first)
        replay, _ = await runner._transcribe_pending_audio_event_once(first)
        assert stt.call_count == 3  # existing pending-merge behavior reprocesses the changed event
    records = first.metadata["audio_transcriptions"]
    assert [(r["source_message_id"], r["source_path"], r["transcript"]) for r in records] == [
        ("voice-1", str(tmp_path / "first.ogg"), "Morning"),
        ("voice-2", str(tmp_path / "second.ogg"), "Evening"),
    ]
    assert text == replay == '"Morning"\n\n"Evening"'
    assert "audio_transcriptions" not in second.metadata
