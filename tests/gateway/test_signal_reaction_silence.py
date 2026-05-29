"""Tests for suppressing no-op replies from Signal reaction wake events."""

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import _should_suppress_signal_reaction_response
from gateway.session import SessionSource


def _event(*, platform=Platform.SIGNAL, raw=None):
    return MessageEvent(
        text="Signal reaction event: Matt reacted 🙏🏼 to a message.",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="+15551234567"),
        raw_message=raw if raw is not None else {"signal_event_type": "reaction"},
    )


def test_suppresses_exact_silent_marker_for_signal_reaction():
    assert _should_suppress_signal_reaction_response(_event(), "[SILENT]") is True


def test_suppresses_legacy_no_response_needed_for_signal_reaction():
    assert _should_suppress_signal_reaction_response(_event(), "No response needed.") is True
    assert _should_suppress_signal_reaction_response(_event(), " no reply needed ") is True


def test_does_not_suppress_actionable_reaction_reply():
    assert _should_suppress_signal_reaction_response(_event(), "All good — I'll do that.") is False


def test_does_not_suppress_non_reaction_signal_message():
    assert _should_suppress_signal_reaction_response(
        _event(raw={"signal_event_type": "message"}),
        "No response needed.",
    ) is False


def test_does_not_suppress_other_platforms():
    assert _should_suppress_signal_reaction_response(
        _event(platform=Platform.TELEGRAM),
        "[SILENT]",
    ) is False
