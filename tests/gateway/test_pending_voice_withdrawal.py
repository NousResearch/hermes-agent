"""Pending voice withdrawal preserves transcript delivery for the remaining clips."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from gateway.platforms.base_pending_merge import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_telegram_voice_v0_regressions import _PendingVoiceAdapter, _runner, _source

pytest_plugins = ("tests.gateway.test_inbound_media_profile_scope",)


def _voice(source, message_id):
    return MessageEvent(
        text="", message_type=MessageType.VOICE, source=source, message_id=message_id,
        media_urls=[f"/cache/{message_id}.ogg"], media_types=["audio/ogg"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawn", ["first", "failed", "last"])
async def test_withdrawal_and_later_merge_echo_each_successful_voice_once(monkeypatch, withdrawn):
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
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio_local_fallback", transcribe)
    pending = adapter._pending_messages[key]
    await runner._transcribe_and_echo_pending_voice(pending, adapter, source, "", log_context="Test")
    found = adapter.withdraw_pending_message(withdrawn, chat_id=source.chat_id, sender_id="alice")
    pending = adapter._pending_messages[key]
    await runner._transcribe_and_echo_pending_voice(pending, adapter, source, "", log_context="Test")
    merge_pending_message_event(adapter._pending_messages, key, _voice(source, "new"))
    await runner._transcribe_and_echo_pending_voice(pending, adapter, source, "", log_context="Test")

    assert (found, adapter.sent) == (
        True, [(source.chat_id, '🎙️ "same words"', {})] * 3,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawn", ["first", "last"])
async def test_routed_voice_snapshot_preserves_echo_receipts_a_b_a(
    two_homes, monkeypatch, withdrawn
):
    from gateway.run import _profile_runtime_scope
    from gateway.run_inbound_media import rehome_inbound_media
    from gateway.shutdown_pending import PendingQueueSnapshot
    from gateway.shutdown_pending_codec import decode_pending_event

    launch, routed = two_homes
    turns = []
    expected = []
    for index, home in enumerate((launch, routed, launch)):
        adapter = _PendingVoiceAdapter()
        runner = _runner(adapter)
        runner._delivery_adapter_for = lambda _source: adapter
        runner._should_echo_stt_transcripts = lambda: True
        source = replace(_source(), user_id="alice")
        events = []
        for name in ("first", "failed", "last"):
            path = launch / "cache" / "audio" / f"{index}-{name}.ogg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            events.append(
                MessageEvent(
                    text="",
                    message_type=MessageType.VOICE,
                    source=source,
                    message_id=name,
                    media_urls=[str(path)],
                    media_types=["audio/ogg"],
                )
            )
        key = adapter._event_session_key(events[0])
        for event in events:
            merge_pending_message_event(adapter._pending_messages, key, event)

        transcribed = []

        def transcribe(path, *args):
            word = Path(path).read_bytes().decode()
            transcribed.append(word)
            if word == "failed":
                return {"success": False, "error": "inaudible"}
            return {"success": True, "transcript": word}

        monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
        monkeypatch.setattr(
            "tools.transcription_tools.transcribe_audio_local_fallback", transcribe
        )
        pending = adapter._pending_messages[key]
        with _profile_runtime_scope(launch):
            await runner._transcribe_and_echo_pending_voice(
                pending, adapter, source, "", log_context="Test"
            )
        with _profile_runtime_scope(home):
            rehome_inbound_media(pending)
            (record,) = PendingQueueSnapshot.capture(key, [pending]).events
            restored = decode_pending_event(json.loads(json.dumps(record)))
            adapter._pending_messages[key] = restored
            await runner._transcribe_and_echo_pending_voice(
                restored, adapter, source, "", log_context="Test"
            )
            transcription_before_withdrawal = list(transcribed)
            adapter.withdraw_pending_message(
                withdrawn, chat_id=source.chat_id, sender_id="alice"
            )
            survivor = adapter._pending_messages[key]
            await runner._transcribe_and_echo_pending_voice(
                survivor, adapter, source, "", log_context="Test"
            )
            path = home / "cache" / "audio" / f"{index}-new.ogg"
            path.write_bytes(b"new")
            merge_pending_message_event(
                adapter._pending_messages,
                key,
                MessageEvent(
                    text="",
                    message_type=MessageType.VOICE,
                    source=source,
                    message_id="new",
                    media_urls=[str(path)],
                    media_types=["audio/ogg"],
                ),
            )
            await runner._transcribe_and_echo_pending_voice(
                survivor, adapter, source, "", log_context="Test"
            )
        paths = [
            str(home / "cache" / "audio" / f"{index}-{name}.ogg")
            for name in ("first", "last")
        ]
        turns.append({
            "echoes": adapter.sent,
            "transcribed_before_withdrawal": transcription_before_withdrawal,
            "snapshot_echoes": record["voice"]["echoed_paths"],
            "snapshot_clips": [clip["path"] for clip in record["voice"]["clips"]],
            "retained_echoes": sorted(
                getattr(survivor, "_gateway_pending_stt_echoed_paths", ())
            ),
            "attachments": [Path(path).read_bytes() for path in survivor.media_urls],
        })
        expected.append({
            "echoes": [
                (source.chat_id, f'🎙️ "{word}"', {}) for word in ("first", "last", "new")
            ],
            "transcribed_before_withdrawal": ["first", "failed", "failed", "last"],
            "snapshot_echoes": sorted(paths),
            "snapshot_clips": paths,
            "retained_echoes": sorted(
                [
                    clip_path
                    for clip_path in paths
                    if not clip_path.endswith(f"-{withdrawn}.ogg")
                ]
                + [str(path)]
            ),
            "attachments": [
                name.encode()
                for name in ("first", "failed", "last", "new")
                if name != withdrawn
            ],
        })
    assert turns == expected
