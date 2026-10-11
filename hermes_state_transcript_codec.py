"""Versioned wire codec for durable session-transcript recovery payloads.

The envelope is deliberately message-shaped rather than column-shaped: it preserves every
JSON field a current or future transcript message carries, while SessionDB remains the one
place that projects those messages onto the canonical schema.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from agent.message_metadata import MESSAGE_UID, message_uid_or_none, stamp_message_uid

SESSION_TRANSCRIPT_SCHEMA = "hermes.session.transcript"
SESSION_TRANSCRIPT_VERSION = 1


class SessionTranscriptCodecError(ValueError):
    """The recovery envelope cannot be decoded without guessing."""


def encode_session_transcript(messages: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Copy *messages* into the current versioned envelope and give every row a stable UID.

    No message-field allowlist lives here. Unknown JSON-compatible fields remain in the spool
    so adding a canonical SessionDB column cannot silently make shutdown recovery lossy again.
    """
    encoded: List[Dict[str, Any]] = []
    seen_uids: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise SessionTranscriptCodecError(f"messages[{index}] must be an object")
        row = dict(message)
        role = row.get("role")
        if not isinstance(role, str) or not role:
            raise SessionTranscriptCodecError(f"messages[{index}].role must be a non-empty string")
        uid = stamp_message_uid(row)
        if uid in seen_uids:
            raise SessionTranscriptCodecError(f"messages[{index}].{MESSAGE_UID} is duplicated")
        seen_uids.add(uid)
        encoded.append(row)
    return {
        "schema": SESSION_TRANSCRIPT_SCHEMA,
        "version": SESSION_TRANSCRIPT_VERSION,
        "messages": encoded,
    }


def decode_session_transcript(envelope: Any) -> List[Dict[str, Any]]:
    """Decode the current envelope without reducing message fields or accepting unknown versions."""
    if not isinstance(envelope, Mapping):
        raise SessionTranscriptCodecError("transcript envelope must be an object")
    if envelope.get("schema") != SESSION_TRANSCRIPT_SCHEMA:
        raise SessionTranscriptCodecError("unrecognized transcript schema")
    if envelope.get("version") != SESSION_TRANSCRIPT_VERSION:
        raise SessionTranscriptCodecError(
            f"unsupported transcript version: {envelope.get('version')!r}")
    messages = envelope.get("messages")
    if not isinstance(messages, list):
        raise SessionTranscriptCodecError("transcript messages must be a list")

    decoded: List[Dict[str, Any]] = []
    seen_uids: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise SessionTranscriptCodecError(f"messages[{index}] must be an object")
        row = dict(message)
        role = row.get("role")
        if not isinstance(role, str) or not role:
            raise SessionTranscriptCodecError(f"messages[{index}].role must be a non-empty string")
        uid = message_uid_or_none(row)
        if uid is None:
            raise SessionTranscriptCodecError(f"messages[{index}].{MESSAGE_UID} is required")
        if uid in seen_uids:
            raise SessionTranscriptCodecError(f"messages[{index}].{MESSAGE_UID} is duplicated")
        seen_uids.add(uid)
        decoded.append(row)
    return decoded
