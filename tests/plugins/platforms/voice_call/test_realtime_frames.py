"""Carrier stream frame adapter goldens (ported from openclaw's
stream-frame-adapter.test.ts vectors)."""

import base64
import json

import pytest

from plugins.platforms.voice_call.realtime.frames import (
    TelnyxStreamFrameAdapter,
    TwilioStreamFrameAdapter,
    adapter_for_provider,
)


def test_adapter_factory():
    assert adapter_for_provider("telnyx").name == "telnyx"
    assert adapter_for_provider("twilio").name == "twilio"
    with pytest.raises(ValueError):
        adapter_for_provider("plivo")


# -- Telnyx -------------------------------------------------------------------


def test_telnyx_parses_start_frame():
    adapter = TelnyxStreamFrameAdapter()
    frame = adapter.parse(json.dumps({
        "event": "start",
        "sequence_number": "1",
        "stream_id": "telnyx-stream-7",
        "start": {
            "call_control_id": "v3:carrier-call-id",
            "call_session_id": "session-1",
            "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1},
        },
    }))
    assert frame.type == "start"
    assert frame.stream_id == "telnyx-stream-7"
    assert frame.provider_call_id == "v3:carrier-call-id"


def test_telnyx_parses_media_mark_stop():
    adapter = TelnyxStreamFrameAdapter()
    frame = adapter.parse(json.dumps({
        "event": "media",
        "stream_id": "telnyx-stream-7",
        "media": {"payload": "AAA=", "timestamp": 40, "track": "inbound_track"},
    }))
    assert frame.type == "media"
    assert frame.payload == base64.b64decode("AAA=")
    assert frame.track == "inbound_track"

    assert adapter.parse(json.dumps(
        {"event": "mark", "mark": {"name": "m1"}})).mark_name == "m1"
    assert adapter.parse(json.dumps({"event": "stop"})).type == "stop"


def test_telnyx_surfaces_error_frames():
    adapter = TelnyxStreamFrameAdapter()
    frame = adapter.parse(json.dumps({
        "event": "error",
        "stream_id": "s",
        "payload": {"code": 100002, "title": "malformed_frame",
                    "detail": "bad payload"},
    }))
    assert frame.type == "error"
    assert "malformed_frame" in frame.error and "100002" in frame.error


def test_telnyx_serializes_outbound_frames():
    adapter = TelnyxStreamFrameAdapter()
    media = json.loads(adapter.serialize_media(b"payload-bytes"))
    assert media == {
        "event": "media",
        "media": {"payload": base64.b64encode(b"payload-bytes").decode()},
    }
    assert json.loads(adapter.serialize_clear()) == {"event": "clear"}
    assert json.loads(adapter.serialize_mark("m")) == {
        "event": "mark", "mark": {"name": "m"},
    }


def test_telnyx_tolerates_garbage():
    adapter = TelnyxStreamFrameAdapter()
    assert adapter.parse("not json").type == "unknown"
    assert adapter.parse('"a string"').type == "unknown"
    assert adapter.parse(json.dumps({"event": "media", "media": {"payload": "!!"}})
                         ).payload == b""


# -- Twilio --------------------------------------------------------------------


def test_twilio_start_captures_stream_sid_for_outbound_envelope():
    adapter = TwilioStreamFrameAdapter()
    frame = adapter.parse(json.dumps({
        "event": "start",
        "streamSid": "MZ123",
        "start": {"callSid": "CA1", "mediaFormat": {"encoding": "audio/x-mulaw"}},
    }))
    assert frame.type == "start"
    assert frame.stream_id == "MZ123"
    assert frame.provider_call_id == "CA1"
    # Outbound frames now carry the captured streamSid.
    media = json.loads(adapter.serialize_media(b"x"))
    assert media["streamSid"] == "MZ123"
    assert json.loads(adapter.serialize_clear())["streamSid"] == "MZ123"


def test_twilio_parses_media_and_connected():
    adapter = TwilioStreamFrameAdapter()
    assert adapter.parse(json.dumps({"event": "connected"})).type == "connected"
    frame = adapter.parse(json.dumps({
        "event": "media", "streamSid": "MZ1",
        "media": {"payload": base64.b64encode(b"\x01\x02").decode(),
                  "track": "inbound"},
    }))
    assert frame.type == "media" and frame.payload == b"\x01\x02"
