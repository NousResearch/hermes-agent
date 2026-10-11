"""Pending voice withdrawal preserves transcript delivery for the remaining clips."""

from dataclasses import replace

import pytest

from gateway.platforms.base_pending_merge import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_telegram_voice_v0_regressions import (
    _PendingVoiceAdapter,
    _runner,
    _source,
)


def _voice(source, message_id):
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        message_id=message_id,
        media_urls=[f"/cache/{message_id}.ogg"],
        media_types=["audio/ogg"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawn", ["first", "failed", "last"])
async def test_withdrawal_and_later_merge_echo_each_successful_voice_once(
    monkeypatch, withdrawn
):
    adapter = _PendingVoiceAdapter()
    runner = _runner(adapter)
    runner._should_echo_stt_transcripts = lambda: True
    source = replace(_source(), user_id="alice")
    events = [_voice(source, name) for name in ("first", "failed", "last")]
    key = adapter._event_session_key(events[0])
    for event in events:
        merge_pending_message_event(adapter._pending_messages, key, event)

    def transcribe(path, *args):
        if path.endswith("/failed.ogg"):
            return {"success": False, "error": "inaudible"}
        return {"success": True, "transcript": "same words"}

    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio_local_fallback", transcribe
    )
    pending = adapter._pending_messages[key]
    await runner._transcribe_and_echo_pending_voice(
        pending, adapter, source, "", log_context="Test"
    )
    found = adapter.withdraw_pending_message(
        withdrawn, chat_id=source.chat_id, sender_id="alice"
    )
    pending = adapter._pending_messages[key]
    await runner._transcribe_and_echo_pending_voice(
        pending, adapter, source, "", log_context="Test"
    )
    merge_pending_message_event(adapter._pending_messages, key, _voice(source, "new"))
    await runner._transcribe_and_echo_pending_voice(
        pending, adapter, source, "", log_context="Test"
    )

    assert (found, adapter.sent) == (
        True,
        [(source.chat_id, '🎙️ "same words"', {})] * 3,
    )
