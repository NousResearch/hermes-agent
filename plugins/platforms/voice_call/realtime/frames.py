"""Carrier media-stream wire-format adapters.

Port of OpenClaw's ``src/webhook/stream-frame-adapter.ts``: each carrier
speaks a slightly different JSON envelope over the media WebSocket. The
bridge consumes only :class:`StreamFrame`.

Telnyx frames (bidirectional RTP streaming):
  start: {event, stream_id, start: {call_control_id, media_format}}
  media: {event, stream_id, media: {payload: b64, track, timestamp}}
  outbound media: {event: "media", media: {payload}}; clear: {event: "clear"}

Twilio Media Streams:
  start: {event, streamSid, start: {callSid, mediaFormat}}
  media: {event, streamSid, media: {payload: b64, track}}
  outbound media adds streamSid; mark/clear also carry streamSid.
"""

import abc
import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StreamFrame:
    type: str                      # "connected" | "start" | "media" | "mark" | "stop" | "error" | "unknown"
    stream_id: Optional[str] = None
    provider_call_id: Optional[str] = None  # carrier call id from the start frame
    payload: bytes = b""           # decoded audio (media frames)
    track: Optional[str] = None
    mark_name: Optional[str] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class StreamFrameAdapter(abc.ABC):
    """Parse inbound carrier WS messages / serialize outbound ones."""

    name: str = ""

    @abc.abstractmethod
    def parse(self, message: str) -> StreamFrame: ...

    @abc.abstractmethod
    def serialize_media(self, payload: bytes) -> str: ...

    @abc.abstractmethod
    def serialize_clear(self) -> str: ...

    @abc.abstractmethod
    def serialize_mark(self, name: str) -> str: ...

    def set_stream_id(self, stream_id: str) -> None:
        """Carriers that envelope outbound frames with a stream id override."""


def _decode_payload(media: Dict[str, Any]) -> bytes:
    payload = media.get("payload")
    if not payload:
        return b""
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return b""


def _parse_json(message: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class TelnyxStreamFrameAdapter(StreamFrameAdapter):
    name = "telnyx"

    def parse(self, message: str) -> StreamFrame:
        data = _parse_json(message)
        if data is None:
            return StreamFrame(type="unknown")
        event = str(data.get("event", ""))
        stream_id = data.get("stream_id")
        if event == "connected":
            return StreamFrame(type="connected", stream_id=stream_id, raw=data)
        if event == "start":
            start = data.get("start") or {}
            return StreamFrame(
                type="start",
                stream_id=stream_id,
                provider_call_id=start.get("call_control_id"),
                raw=data,
            )
        if event == "media":
            media = data.get("media") or {}
            return StreamFrame(
                type="media",
                stream_id=stream_id,
                payload=_decode_payload(media),
                track=media.get("track"),
                raw=data,
            )
        if event == "mark":
            return StreamFrame(
                type="mark", stream_id=stream_id,
                mark_name=(data.get("mark") or {}).get("name"), raw=data,
            )
        if event == "stop":
            return StreamFrame(type="stop", stream_id=stream_id, raw=data)
        if event == "error":
            payload = data.get("payload") or {}
            detail = " ".join(
                str(payload.get(k)) for k in ("code", "title", "detail")
                if payload.get(k)
            )
            return StreamFrame(
                type="error", stream_id=stream_id,
                error=detail or "unknown stream error", raw=data,
            )
        return StreamFrame(type="unknown", stream_id=stream_id, raw=data)

    def serialize_media(self, payload: bytes) -> str:
        return json.dumps(
            {"event": "media",
             "media": {"payload": base64.b64encode(payload).decode()}}
        )

    def serialize_clear(self) -> str:
        return json.dumps({"event": "clear"})

    def serialize_mark(self, name: str) -> str:
        return json.dumps({"event": "mark", "mark": {"name": name}})


class TwilioStreamFrameAdapter(StreamFrameAdapter):
    name = "twilio"

    def __init__(self):
        self._stream_sid: Optional[str] = None

    def set_stream_id(self, stream_id: str) -> None:
        self._stream_sid = stream_id

    def parse(self, message: str) -> StreamFrame:
        data = _parse_json(message)
        if data is None:
            return StreamFrame(type="unknown")
        event = str(data.get("event", ""))
        stream_sid = data.get("streamSid")
        if event == "connected":
            return StreamFrame(type="connected", raw=data)
        if event == "start":
            start = data.get("start") or {}
            if stream_sid:
                self._stream_sid = stream_sid
            return StreamFrame(
                type="start",
                stream_id=stream_sid,
                provider_call_id=start.get("callSid"),
                raw=data,
            )
        if event == "media":
            media = data.get("media") or {}
            return StreamFrame(
                type="media",
                stream_id=stream_sid,
                payload=_decode_payload(media),
                track=media.get("track"),
                raw=data,
            )
        if event == "mark":
            return StreamFrame(
                type="mark", stream_id=stream_sid,
                mark_name=(data.get("mark") or {}).get("name"), raw=data,
            )
        if event == "stop":
            return StreamFrame(type="stop", stream_id=stream_sid, raw=data)
        return StreamFrame(type="unknown", stream_id=stream_sid, raw=data)

    def _envelope(self, body: Dict[str, Any]) -> str:
        if self._stream_sid:
            body["streamSid"] = self._stream_sid
        return json.dumps(body)

    def serialize_media(self, payload: bytes) -> str:
        return self._envelope(
            {"event": "media",
             "media": {"payload": base64.b64encode(payload).decode()}}
        )

    def serialize_clear(self) -> str:
        return self._envelope({"event": "clear"})

    def serialize_mark(self, name: str) -> str:
        return self._envelope({"event": "mark", "mark": {"name": name}})


def adapter_for_provider(provider_name: str) -> StreamFrameAdapter:
    if provider_name == "telnyx":
        return TelnyxStreamFrameAdapter()
    if provider_name == "twilio":
        return TwilioStreamFrameAdapter()
    raise ValueError(f"no stream frame adapter for provider {provider_name!r}")
