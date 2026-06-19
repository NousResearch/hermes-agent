"""Bridge wire protocol — Python mirror of the .NET worker's ``Protocol.cs``.

The companion media worker (AzureBot / OpenClawBridge) and this driver exchange
newline-free JSON text frames over one WebSocket per call, discriminated on a
``type`` field, camelCase keys. This module models the **inbound** messages
(worker -> gateway) as dataclasses with a single :func:`decode` entry point, and
provides **outbound** builders (gateway -> worker).

The contract is fixed by the existing worker; keep field names camelCase and
additive — unknown fields are ignored and unknown message types degrade
gracefully (older/newer peers interoperate). See ``Protocol.cs`` for the
authoritative C# records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# ── Message type discriminators ──────────────────────────────────────────────

# Inbound: worker -> gateway
TYPE_SESSION_START = "session.start"
TYPE_SESSION_END = "session.end"
TYPE_RECORDING_STATUS = "recording.status"
TYPE_AUDIO_FRAME = "audio.frame"  # also outbound (TTS)
TYPE_VIDEO_FRAME = "video.frame"
TYPE_PARTICIPANTS = "participants"
TYPE_DTMF = "dtmf"
TYPE_PING = "ping"

# Outbound: gateway -> worker
TYPE_EXPRESSION = "expression"
TYPE_SPEECH_MARKS = "speech.marks"
TYPE_DISPLAY_IMAGE = "display.image"
TYPE_ASSISTANT_CANCEL = "assistant.cancel"
TYPE_PONG = "pong"


class ProtocolError(ValueError):
    """Raised when an inbound frame is malformed or fails schema validation."""


# ── Inbound message models ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CallerInfo:
    """Caller identity from ``session.start``. All fields best-effort/nullable.

    Blank/whitespace strings are coerced to ``None`` so distinct anonymous
    callers never collide on an empty AAD id (cross-caller memory bleed guard).
    """

    aad_id: Optional[str] = None
    display_name: Optional[str] = None
    tenant_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CallerInfo":
        payload = payload or {}
        return cls(
            aad_id=_clean(payload.get("aadId")),
            display_name=_clean(payload.get("displayName")),
            tenant_id=_clean(payload.get("tenantId")),
        )


@dataclass(frozen=True)
class SessionStart:
    type: str
    call_id: str
    thread_id: str
    caller: CallerInfo
    recording_status: Optional[str] = None  # "active" | "inactive" | "unknown"
    direction: Optional[str] = None  # "inbound" | "outbound"


@dataclass(frozen=True)
class SessionEnd:
    type: str
    reason: str


@dataclass(frozen=True)
class RecordingStatus:
    type: str
    status: str  # "active" | "inactive" | "unknown"


@dataclass(frozen=True)
class AudioFrame:
    type: str
    seq: int
    timestamp_ms: int
    payload_base64: str
    speaker_name: Optional[str] = None  # unmixed-audio attribution (additive)


@dataclass(frozen=True)
class VideoFrame:
    type: str
    source: str  # "camera" | "screenshare"
    ts: int
    width: int
    height: int
    mime: str
    data_base64: str
    participant_id: Optional[str] = None
    participant_name: Optional[str] = None


@dataclass(frozen=True)
class Participants:
    type: str
    count: int  # human participants, excludes the bot


@dataclass(frozen=True)
class Dtmf:
    type: str
    digit: str  # "0"-"9", "*", "#"


@dataclass(frozen=True)
class Ping:
    type: str
    ts: int


InboundMessage = (
    SessionStart
    | SessionEnd
    | RecordingStatus
    | AudioFrame
    | VideoFrame
    | Participants
    | Dtmf
    | Ping
)


def decode(raw: str | bytes) -> InboundMessage:
    """Parse one inbound text frame into a typed message.

    Raises :class:`ProtocolError` on malformed JSON, a missing/blank ``type``,
    or a required field missing for a known type. An unknown ``type`` also raises
    so the caller can log-and-skip without crashing the read loop.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("frame is not a JSON object")

    mtype = obj.get("type")
    if not isinstance(mtype, str) or not mtype:
        raise ProtocolError("missing 'type'")

    try:
        if mtype == TYPE_SESSION_START:
            return SessionStart(
                type=mtype,
                call_id=_require_str(obj, "callId"),
                thread_id=_require_str(obj, "threadId"),
                caller=CallerInfo.from_dict(obj.get("caller")),
                recording_status=_clean(obj.get("recordingStatus")),
                direction=_clean(obj.get("direction")),
            )
        if mtype == TYPE_SESSION_END:
            return SessionEnd(type=mtype, reason=str(obj.get("reason") or ""))
        if mtype == TYPE_RECORDING_STATUS:
            return RecordingStatus(type=mtype, status=_require_str(obj, "status"))
        if mtype == TYPE_AUDIO_FRAME:
            return AudioFrame(
                type=mtype,
                seq=int(obj.get("seq") or 0),
                timestamp_ms=int(obj.get("timestampMs") or 0),
                payload_base64=_require_str(obj, "payloadBase64"),
                speaker_name=_clean(obj.get("speakerName")),
            )
        if mtype == TYPE_VIDEO_FRAME:
            return VideoFrame(
                type=mtype,
                source=_require_str(obj, "source"),
                ts=int(obj.get("ts") or 0),
                width=int(obj.get("width") or 0),
                height=int(obj.get("height") or 0),
                mime=str(obj.get("mime") or "image/jpeg"),
                data_base64=_require_str(obj, "dataBase64"),
                participant_id=_clean(obj.get("participantId")),
                participant_name=_clean(obj.get("participantName")),
            )
        if mtype == TYPE_PARTICIPANTS:
            return Participants(type=mtype, count=int(obj.get("count") or 0))
        if mtype == TYPE_DTMF:
            return Dtmf(type=mtype, digit=_require_str(obj, "digit"))
        if mtype == TYPE_PING:
            return Ping(type=mtype, ts=int(obj.get("ts") or 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"bad {mtype!r} frame: {exc}") from exc

    raise ProtocolError(f"unknown message type: {mtype!r}")


# ── Outbound builders (gateway -> worker) ────────────────────────────────────


def audio_frame(seq: int, timestamp_ms: int, payload_base64: str) -> dict[str, Any]:
    """Outbound TTS / realtime audio chunk (PCM 16 kHz, base64, 20 ms)."""
    return {
        "type": TYPE_AUDIO_FRAME,
        "seq": seq,
        "timestampMs": timestamp_ms,
        "payloadBase64": payload_base64,
    }


def expression(emotion: str) -> dict[str, Any]:
    """Avatar emotion cue. Best-effort/cosmetic; older worker ignores it."""
    return {"type": TYPE_EXPRESSION, "emotion": emotion}


def speech_marks(marks: list[dict[str, int]], ts: int = 0) -> dict[str, Any]:
    """Viseme timeline for lip-sync. ``marks`` = ``[{"tMs": int, "visemeId": int}]``."""
    return {"type": TYPE_SPEECH_MARKS, "ts": ts, "marks": marks}


def display_image(
    data_base64: str,
    mime: str,
    *,
    duration_ms: int | None = None,
    mode: str | None = None,  # "fullscreen" (default) | "overlay" (PiP)
    caption: str | None = None,
    ts: int = 0,
) -> dict[str, Any]:
    """``show_to_caller`` — render an image on the bot's tile, then return to avatar."""
    msg: dict[str, Any] = {
        "type": TYPE_DISPLAY_IMAGE,
        "dataBase64": data_base64,
        "mime": mime,
        "ts": ts,
    }
    if duration_ms is not None:
        msg["durationMs"] = duration_ms
    if mode is not None:
        msg["mode"] = mode
    if caption is not None:
        msg["caption"] = caption
    return msg


def assistant_cancel(turn_id: int) -> dict[str, Any]:
    """Barge-in — tell the worker to flush playback for ``turn_id``."""
    return {"type": TYPE_ASSISTANT_CANCEL, "turnId": turn_id}


def pong(ts: int) -> dict[str, Any]:
    """Keepalive reply echoing the worker's ping timestamp."""
    return {"type": TYPE_PONG, "ts": ts}


def encode(msg: dict[str, Any]) -> str:
    """Serialize an outbound message dict to a compact JSON text frame."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)


# ── helpers ──────────────────────────────────────────────────────────────────


def _clean(value: Any) -> Optional[str]:
    """Normalize to a non-blank string or ``None`` (blank-as-null guard)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if value is None or str(value).strip() == "":
        raise ProtocolError(f"missing required field {key!r}")
    return str(value)
