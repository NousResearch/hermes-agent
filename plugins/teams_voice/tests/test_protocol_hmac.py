"""Tests for the bridge protocol decode/encode and HMAC handshake."""

from __future__ import annotations

import json

import pytest

from plugins.teams_voice import hmac_auth, protocol


def test_decode_session_start_blank_aad_is_none():
    raw = json.dumps(
        {
            "type": "session.start",
            "callId": "abc",
            "threadId": "t1",
            "caller": {"aadId": "   ", "displayName": "Alaa"},
            "direction": "inbound",
            "recordingStatus": "inactive",
        }
    )
    msg = protocol.decode(raw)
    assert isinstance(msg, protocol.SessionStart)
    assert msg.call_id == "abc"
    assert msg.caller.aad_id is None  # blank coerced to None
    assert msg.caller.display_name == "Alaa"


def test_decode_audio_frame_and_missing_field():
    msg = protocol.decode(
        json.dumps({"type": "audio.frame", "seq": 3, "timestampMs": 60, "payloadBase64": "AAAA"})
    )
    assert isinstance(msg, protocol.AudioFrame)
    assert msg.seq == 3 and msg.payload_base64 == "AAAA"

    with pytest.raises(protocol.ProtocolError):
        protocol.decode(json.dumps({"type": "audio.frame", "seq": 1}))


def test_decode_rejects_unknown_and_malformed():
    with pytest.raises(protocol.ProtocolError):
        protocol.decode("{not json")
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(json.dumps({"type": "bogus.message"}))


def test_outbound_builders_camelcase():
    assert protocol.expression("happy") == {"type": "expression", "emotion": "happy"}
    marks = protocol.speech_marks([{"tMs": 0, "visemeId": 2}], ts=0)
    assert marks["marks"][0]["visemeId"] == 2
    img = protocol.display_image("ZZ", "image/png", mode="overlay", caption="hi")
    assert img["mode"] == "overlay" and img["caption"] == "hi"
    assert protocol.assistant_cancel(7) == {"type": "assistant.cancel", "turnId": 7}


def test_hmac_sign_matches_dotnet_payload_shape():
    # payload is "{timestampMs}.{callId}", HMAC-SHA256, lowercase hex
    sig = hmac_auth.sign("secret", 1700000000000, "call-1")
    assert sig == sig.lower() and len(sig) == 64


def test_hmac_verify_window_and_signature():
    secret, call_id, now = "s3cr3t", "c1", 1_000_000
    ts = now
    sig = hmac_auth.sign(secret, ts, call_id)

    ok, reason = hmac_auth.verify_upgrade(
        secret=secret, call_id=call_id, timestamp_header=str(ts),
        signature_header=sig, window_ms=60_000, now_ms=now,
    )
    assert ok, reason

    # bad signature
    ok, reason = hmac_auth.verify_upgrade(
        secret=secret, call_id=call_id, timestamp_header=str(ts),
        signature_header="deadbeef", window_ms=60_000, now_ms=now,
    )
    assert not ok and reason == "bad signature"

    # outside window
    ok, reason = hmac_auth.verify_upgrade(
        secret=secret, call_id=call_id, timestamp_header=str(ts - 120_000),
        signature_header=hmac_auth.sign(secret, ts - 120_000, call_id),
        window_ms=60_000, now_ms=now,
    )
    assert not ok and reason == "timestamp outside window"


def test_hmac_replay_guard_single_use():
    guard = hmac_auth.ReplayGuard(window_ms=60_000)
    ts = hmac_auth._now_ms()  # realistic timestamp so the entry isn't pruned
    assert guard.check_and_record("c1", ts, "sigA") is True
    assert guard.check_and_record("c1", ts, "sigA") is False  # replay rejected
    assert guard.check_and_record("c1", ts + 1, "sigB") is True  # fresh ts ok
