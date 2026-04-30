"""Tests for LINE webhook signature verification and payload parsing."""
import json

import pytest

from gateway.platforms.line import verify_signature, parse_events
from tests.gateway.conftest import line_sign as _sign


@pytest.mark.parametrize(
    "body, signed, secret, expected",
    [
        # Valid signature on a normal payload.
        (b'{"events":[]}', True, "test_secret", True),
        # Valid signature on empty body — boundary case.
        (b"", True, "test_secret", True),
        # Invalid signature — must be rejected.
        (b'{"events":[]}', False, "test_secret", False),
        # Empty signature on empty body — must be rejected.
        (b"", False, "test_secret", False),
        # Outbound-only mode (LINE_CHANNEL_SECRET unset) — every inbound
        # rejected so the gateway can't accept un-verifiable traffic.
        (b'{"events": []}', False, "", False),
    ],
    ids=["valid", "valid_empty_body", "invalid_sig", "empty_sig_empty_body", "outbound_only_no_secret"],
)
def test_verify_signature_edge_cases(body, signed, secret, expected):
    sig = _sign(secret, body) if signed and secret else "not-the-real-signature" if not signed else "any-sig"
    assert verify_signature(body, sig, secret) is expected


def test_parse_events_returns_empty_for_no_events():
    body = json.dumps({"destination": "U1", "events": []}).encode()
    assert parse_events(body) == []


def test_parse_events_extracts_message_event():
    payload = {
        "destination": "U1",
        "events": [
            {
                "type": "message",
                "replyToken": "rt-1",
                "source": {"type": "user", "userId": "Uabc"},
                "timestamp": 1234567890,
                "message": {"id": "m1", "type": "text", "text": "hi"},
            }
        ],
    }
    body = json.dumps(payload).encode()
    events = parse_events(body)
    assert len(events) == 1
    assert events[0]["type"] == "message"
    assert events[0]["source"]["userId"] == "Uabc"


def test_parse_events_returns_empty_for_malformed_json():
    assert parse_events(b"not json") == []


def test_parse_events_returns_empty_for_non_dict_payload():
    assert parse_events(b"[1,2,3]") == []
