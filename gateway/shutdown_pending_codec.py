"""Data-only attribution and voice receipts for pending event reconstruction."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

from gateway.config import Platform
from gateway.pending_native import PendingNativeInput
from gateway.platforms.base_pending import Merge
from gateway.platforms import base_pending_merge
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class PendingMerge(Enum):
    MEDIA = "pending_media"
    TEXT = "pending_text"
    BATCHED = "batched_text"
    DEBOUNCED = "debounced_text"


_MERGERS: dict[PendingMerge, Merge] = {
    PendingMerge.MEDIA: base_pending_merge._absorb_pending_media,
    PendingMerge.TEXT: base_pending_merge._absorb_pending_text,
    PendingMerge.BATCHED: base_pending_merge._append_batched_text,
    PendingMerge.DEBOUNCED: base_pending_merge._append_debounced_text,
}
_OPERATIONS = {merge: operation for operation, merge in _MERGERS.items()}


def _file_digest(path: str) -> str | None:
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def capture_pending_provenance(event: MessageEvent) -> dict[str, Any]:
    from gateway.shutdown_pending import _capture_event

    from gateway.input_owner import capture_gateway_input_owner

    recorded: dict[str, Any] = {"input_owner": capture_gateway_input_owner(event)}
    native = event._pending_native_input
    if native is None:
        from gateway.session_identity import identity_of

        identity = identity_of(event.source)
        adapter = identity.adapter() if identity is not None else None
        capture = getattr(adapter, "pending_native_input", None)
        native = capture(event) if callable(capture) else None
    if native is not None:
        recorded["native"] = native.to_payload()
    if event._merged_parts:
        recorded["attribution"] = [
            {"operation": _OPERATIONS[merge].value if merge is not None else None,
             "record": _capture_event(part)}
            for part, merge in event._merged_parts
        ]
    clips = getattr(event, "_gateway_pending_stt_clips", ())
    transcription = getattr(event, "_gateway_pending_stt_input", None)
    text = getattr(event, "_gateway_pending_stt_text", None)
    echoed = set(getattr(event, "_gateway_pending_stt_echoed_paths", ()))
    if clips or echoed or transcription is not None or text is not None:
        recorded["voice"] = {
            "text": text,
            "clips": [{"path": clip.path, "text": clip.text, "digest": _file_digest(clip.path)}
                      for clip in clips],
            "echoed_paths": sorted(echoed.intersection(event.media_urls)),
            "attachments": [{"path": path, "digest": _file_digest(path)} for path in event.media_urls],
        }
        if transcription is not None:
            recorded["voice"]["parts"] = [
                {"text": part.text, "clip_path": part.clip.path if part.clip is not None else None}
                for part in transcription.parts
            ]
    return recorded


def _matches_type(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        return any(_matches_type(value, arm) for arm in args)
    if origin is list:
        return isinstance(value, list) and all(_matches_type(item, args[0]) for item in value)
    if origin is dict:
        return isinstance(value, dict) and all(_matches_type(key, args[0]) and _matches_type(item, args[1])
                                               for key, item in value.items())
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, annotation)


def _decode_fields(body: dict[str, Any], cls: type[MessageEvent] | type[SessionSource]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls) if item.init and not item.name.startswith("_")}
    if set(body) - allowed:
        raise ValueError("pending event contains unsupported fields")
    annotations = cls.__annotations__
    if any(not _matches_type(value, annotations[key]) for key, value in body.items()):
        raise ValueError("pending event field has an invalid type")
    return body


def decode_pending_source(record: dict[str, Any]) -> SessionSource:
    body = record.get("event")
    if not isinstance(body, dict) or not isinstance(body.get("source"), dict):
        raise ValueError("pending event requires a source")
    source_body = dict(body["source"])
    source_body["platform"] = Platform(source_body["platform"])
    return SessionSource(**_decode_fields(source_body, SessionSource))


def decode_pending_event(record: dict[str, Any], *, adapter: Any = None) -> MessageEvent:
    uid = record.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("pending event requires its stable identity")
    source = decode_pending_source(record)
    body = record["event"]
    event_body = {key: value for key, value in body.items() if key != "source"}
    event_body["message_type"] = MessageType(event_body["message_type"])
    event_body["timestamp"] = datetime.fromisoformat(event_body["timestamp"])
    event_body["source"] = source
    event = MessageEvent(**_decode_fields(event_body, MessageEvent))
    setattr(event, "_pending_snapshot_uid", uid)
    if "native" in record:
        event._pending_native_input = PendingNativeInput.from_payload(record["native"], event)
    if "input_owner" in record:
        from gateway.input_owner import restore_gateway_input_owner
        restore_gateway_input_owner(event, record["input_owner"])
    attribution = record.get("attribution")
    if attribution is not None:
        if not isinstance(attribution, list) or len(attribution) < 2:
            raise ValueError("pending attribution requires its original contributions")
        for index, part in enumerate(attribution):
            if not isinstance(part, dict) or not isinstance(part.get("record"), dict):
                raise ValueError("pending contribution must be an object")
            operation = part.get("operation")
            if index == 0 and operation is not None or index > 0 and operation is None:
                raise ValueError("pending contribution merge order is invalid")
            merge = _MERGERS[PendingMerge(operation)] if operation is not None else None
            event._merged_parts.append((decode_pending_event(part["record"], adapter=adapter), merge))
    elif event.merged_message_ids:
        raise ValueError("legacy merged input has no withdrawal provenance")
    context = record.get("context")
    if context is not None:
        if adapter is None:
            raise ValueError("pending external context requires its current adapter")
        from plugins.platforms.matrix.pending_context import restore_pending_context
        restore_pending_context(adapter, event, context)
    voice = record.get("voice")
    if voice is not None:
        _restore_voice(event, voice)
    return event


def _restore_voice(event: MessageEvent, voice: dict[str, Any]) -> None:
    from gateway.run_inbound_voice import VoiceClipTranscript, VoiceTranscription, VoiceTranscriptPart

    if not isinstance(voice, dict) or not isinstance(voice.get("clips"), list):
        raise ValueError("pending voice receipts must be an object with clips")
    echoed, text = voice.get("echoed_paths"), voice.get("text")
    if (not isinstance(echoed, list) or any(not isinstance(path, str) for path in echoed)
            or text is not None and not isinstance(text, str)):
        raise ValueError("pending voice receipt fields are invalid")
    clips = []
    for clip in voice["clips"]:
        if (not isinstance(clip, dict) or not isinstance(clip.get("path"), str)
                or not isinstance(clip.get("text"), str)
                or clip.get("digest") is not None and not isinstance(clip["digest"], str)):
            raise ValueError("pending voice clip receipt is invalid")
        path = clip["path"]
        if path not in event.media_urls or not clip["digest"] or _file_digest(path) != clip["digest"]:
            continue
        clips.append(VoiceClipTranscript(path, clip["text"]))
    valid_paths = {clip.path for clip in clips}
    setattr(event, "_gateway_pending_stt_echoed_paths", set(echoed).intersection(valid_paths))
    attachments = voice.get("attachments")
    if (not isinstance(attachments, list)
            or any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
                   or item.get("digest") is not None and not isinstance(item["digest"], str)
                   for item in attachments)):
        raise ValueError("pending voice attachment receipts are invalid")
    unchanged = ([item["path"] for item in attachments] == event.media_urls and all(
        item["digest"] is not None and _file_digest(item["path"]) == item["digest"]
        for item in attachments
    ))
    if unchanged and len(clips) == len(voice["clips"]) and text is not None:
        setattr(event, "_gateway_pending_stt_text", text)
        setattr(event, "_gateway_pending_stt_clips", tuple(clips))
        setattr(event, "_gateway_pending_stt_transcripts", [clip.text for clip in clips])
        records = voice.get("parts")
        if records is None:
            return
        if not isinstance(records, list):
            raise ValueError("pending voice parts must be a list")
        by_path = {clip.path: clip for clip in clips}
        parts = []
        for record in records:
            if (not isinstance(record, dict) or not isinstance(record.get("text"), str)
                    or record.get("clip_path") is not None and not isinstance(record["clip_path"], str)):
                raise ValueError("pending voice part receipt is invalid")
            path = record.get("clip_path")
            if path is not None and path not in by_path:
                raise ValueError("pending voice part has no successful clip receipt")
            parts.append(VoiceTranscriptPart(record["text"], by_path.get(path)))
        setattr(event, "_gateway_pending_stt_input", VoiceTranscription(text, tuple(clips), tuple(parts)))
