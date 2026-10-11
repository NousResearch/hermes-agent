"""Durable pending input retains the provenance needed for withdrawal."""

from dataclasses import fields
import hashlib
import json

import pytest

from gateway.config import Platform
from gateway.platforms.base_pending import merge_recorded, withdraw_from_event
from gateway.platforms.base_pending_merge import (
    _absorb_pending_media, _absorb_pending_text,
    _append_batched_text, _append_debounced_text,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_inbound_voice import VoiceClipTranscript
from gateway.session import SessionSource
from gateway.shutdown_pending import PendingQueueSnapshot


@pytest.mark.parametrize("operation,merge", [
    ("pending_media", _absorb_pending_media), ("pending_text", _absorb_pending_text),
    ("batched_text", _append_batched_text), ("debounced_text", _append_debounced_text),
])
def test_snapshot_reconstructs_withdrawable_voice_parts_and_echo_receipts(tmp_path, monkeypatch, operation, merge):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    paths = [tmp_path / "first.ogg", tmp_path / "second.ogg"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="sender")
    has_voice = operation == "pending_media"
    events = [MessageEvent(text=f"caption {index}", source=source, message_id=str(index),
                           message_type=MessageType.VOICE if has_voice else MessageType.TEXT,
                           media_urls=[str(path)] if has_voice else [],
                           media_types=["audio/ogg"] if has_voice else [], channel_context=f"history {index}")
              for index, path in enumerate(paths)]
    first, second = events
    merge_recorded(first, second, merge)
    if has_voice:
        cached_text = '"first"\n\n"second"\n\n' + first.text
        setattr(first, "_gateway_pending_stt_text", cached_text)
        setattr(first, "_gateway_pending_stt_transcripts", ["first", "second"])
        clips = tuple(VoiceClipTranscript(str(path), word)
                      for path, word in zip(paths, ("first", "second")))
        setattr(first, "_gateway_pending_stt_clips", clips)
        setattr(first, "_gateway_pending_stt_echoed_paths", {str(paths[0])})
    captured = PendingQueueSnapshot.capture("key", [first])
    record, = captured.events
    assert [(part["operation"], part["record"]["event"]["message_id"])
            for part in record.get("attribution", [])] == [(None, "0"), (operation, "1")]
    if has_voice:
        assert record["voice"] == {
            "text": getattr(first, "_gateway_pending_stt_text"),
            "clips": [{"path": str(path), "text": word,
                       "digest": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path, word in zip(paths, ("first", "second"))],
            "echoed_paths": [str(paths[0])],
            "attachments": [{"path": str(path), "digest": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in paths],
        }

    from gateway.shutdown_pending_codec import decode_pending_event
    restored = decode_pending_event(json.loads(json.dumps(record)))
    remaining_before = withdraw_from_event(first, lambda part: part.message_id == "0")
    remaining_after = withdraw_from_event(restored, lambda part: part.message_id == "0")
    before, after = remaining_before[1], remaining_after[1]
    public = lambda event: {item.name: getattr(event, item.name) for item in fields(event) if item.init}
    assert (remaining_after[0], public(after), getattr(after, "_gateway_pending_stt_echoed_paths", set())) == (
        remaining_before[0], public(before), set(),
    )
    assert (getattr(restored, "_pending_snapshot_uid"), getattr(after, "_pending_snapshot_uid", None)) == (
        record["uid"], record["uid"],
    )
    if has_voice:
        assert (getattr(restored, "_gateway_pending_stt_text"), getattr(restored, "_gateway_pending_stt_clips"),
                getattr(restored, "_gateway_pending_stt_echoed_paths")) == (
            getattr(first, "_gateway_pending_stt_text"), getattr(first, "_gateway_pending_stt_clips"), {str(paths[0])},
        )


@pytest.mark.parametrize("change", [
    "unknown-operation", "field-type", "changed-file", "missing-file",
    "voice-reference-outside", "voice-reference-empty", "voice-reference-label",
])
def test_reconstruction_rejects_invalid_records_and_stale_voice_cache(tmp_path, monkeypatch, change):
    from gateway.shutdown_pending_codec import decode_pending_event

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"original")
    failed_path = tmp_path / "failed.ogg"
    failed_path.write_bytes(b"untranscribed")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="sender")
    first = MessageEvent(text="caption", source=source, message_id="one", message_type=MessageType.VOICE,
                         media_urls=[str(path), str(failed_path)], media_types=["audio/ogg", "audio/ogg"])
    second = MessageEvent(text="later", source=source, message_id="two")
    merge_recorded(first, second, _absorb_pending_media)
    setattr(first, "_gateway_pending_stt_text", '"voice"\n\ncaption\n\nlater')
    setattr(first, "_gateway_pending_stt_clips", (VoiceClipTranscript(str(path), "voice"),))
    setattr(first, "_gateway_pending_stt_echoed_paths", {str(path)})
    record, = PendingQueueSnapshot.capture("key", [first]).events
    invalid_references = {
        "voice-reference-outside": {"path": str(tmp_path / "unrelated.ogg"), "rendered_path": "generated"},
        "voice-reference-empty": {"path": str(failed_path), "rendered_path": ""},
        "voice-reference-label": {"path": str(failed_path), "rendered_path": "absent"},
    }
    if change in invalid_references:
        record["voice"]["parts"] = [{
            "text": "generated", "clip_path": None,
            "attachment": invalid_references[change],
        }]
        with pytest.raises(ValueError):
            decode_pending_event(record)
        return
    if change == "unknown-operation":
        record["attribution"][1]["operation"] = "arbitrary.module.call"
    elif change == "field-type":
        record["event"]["allow_gateway_control"] = "true"
    elif change == "changed-file":
        failed_path.write_bytes(b"replacement")
    else:
        failed_path.unlink()
    if change in {"unknown-operation", "field-type"}:
        with pytest.raises(ValueError):
            decode_pending_event(record)
        return
    restored = decode_pending_event(record)
    assert (getattr(restored, "_gateway_pending_stt_text", None),
            getattr(restored, "_gateway_pending_stt_clips", ()),
            getattr(restored, "_gateway_pending_stt_echoed_paths", set())) == (None, (), {str(path)})


@pytest.mark.parametrize(
    "cached_text",
    ["Opaque speech @file:mine.txt\n\nGenerated path @file:planted.txt", ""],
    ids=["opaque", "empty"],
)
@pytest.mark.parametrize("change", ["unchanged", "changed", "missing"])
def test_snapshot_preserves_opaque_voice_only_while_attachments_match(
    tmp_path, monkeypatch, cached_text, change
):
    from gateway.shutdown_pending_codec import decode_pending_event

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    paths = [tmp_path / "first.ogg", tmp_path / "second.ogg"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]))
    event = MessageEvent(
        text="Caption @file:caption.txt",
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", user_id="sender"
        ),
        message_id="voice",
        message_type=MessageType.VOICE,
        media_urls=[str(path) for path in paths],
        media_types=["audio/ogg", "audio/ogg"],
    )
    setattr(event, "_gateway_pending_stt_text", cached_text)
    (record,) = PendingQueueSnapshot.capture("key", [event]).events
    if change == "changed":
        paths[1].write_bytes(b"replacement")
    if change == "missing":
        paths[1].unlink()

    restored = decode_pending_event(json.loads(json.dumps(record)))
    assert {
        "text": getattr(restored, "_gateway_pending_stt_text", None),
        "clips": getattr(restored, "_gateway_pending_stt_clips", ()),
        "transcripts": getattr(restored, "_gateway_pending_stt_transcripts", None),
        "echoes": getattr(restored, "_gateway_pending_stt_echoed_paths", set()),
    } == {
        "text": cached_text if change == "unchanged" else None,
        "clips": (),
        "transcripts": [] if change == "unchanged" else None,
        "echoes": set(),
    }
