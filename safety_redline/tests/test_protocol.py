"""Unit tests for ``safety_redline.protocol``."""

from __future__ import annotations

from safety_redline import SafetyRedline, SafetyRedlineProtocol, SafetyState


def _proto():
    return SafetyRedlineProtocol(redline=SafetyRedline())


def test_handle_failure_emits_paused_event():
    proto = _proto()
    proto.handle({"type": "safety.report_failure", "body": {"reason": "timeout"}})
    proto.handle({"type": "safety.report_failure", "body": {}})
    event = proto.handle({"type": "safety.report_failure", "body": {}})
    assert event is not None
    assert event.state is SafetyState.PAUSED


def test_handle_success_resets_streak():
    proto = _proto()
    proto.handle({"type": "safety.report_failure", "body": {"reason": "x"}})
    proto.handle({"type": "safety.report_failure", "body": {"reason": "x"}})
    proto.handle({"type": "safety.report_success", "body": {}})
    assert proto.redline._failure_streak == 0


def test_handle_reset_clears_state():
    proto = _proto()
    for _ in range(3):
        proto.handle({"type": "safety.report_failure", "body": {}})
    assert proto.redline.state is SafetyState.PAUSED
    proto.handle({"type": "safety.reset", "body": {"operator": "test"}})
    assert proto.redline.state is SafetyState.HEALTHY


def test_snapshot_round_trip():
    proto = _proto()
    proto.handle({"type": "safety.report_failure", "body": {}})
    snap = proto.snapshot()
    assert snap["state"] in ("healthy", "warn")
    assert snap["failure_streak"] == 1


def test_decode_and_encode_round_trip():
    proto = _proto()
    raw = proto.encode({"type": "safety.report_failure", "body": {"reason": "boom"}})
    decoded = proto.decode(raw)
    assert decoded["type"] == "safety.report_failure"
    assert decoded["body"]["reason"] == "boom"


def test_decode_empty_message_raises():
    proto = _proto()
    try:
        proto.decode(b"\n")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty input")


def test_unknown_message_type_returns_none():
    proto = _proto()
    event = proto.handle({"type": "complete.unknown", "body": {}})
    assert event is None
