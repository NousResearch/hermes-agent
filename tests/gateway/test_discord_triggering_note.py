"""Regression for #71304: Discord triggering-message note must not be persisted."""

import re

_PATTERN = re.compile(
    r"^\[Triggering message id: `[^`]+` . use as `message_id` for reply/react/pin via the discord tools.\]\s*\n*"
)


def test_triggering_note_stripped():
    raw = "[Triggering message id: `1234567890` \u2014 use as `message_id` for reply/react/pin via the discord tools.]\n\nHello, how are you?"
    cleaned = _PATTERN.sub("", raw, count=1).lstrip()
    assert cleaned == "Hello, how are you?", f"Got: {cleaned!r}"
    assert "Triggering" not in cleaned


def test_normal_message_unchanged():
    raw = "Just a normal message."
    cleaned = _PATTERN.sub("", raw, count=1).lstrip()
    assert cleaned == raw


def test_note_with_extra_newlines():
    raw = "[Triggering message id: `msg_001` \u2014 use as `message_id` for reply/react/pin via the discord tools.]\n\n\nUser text"
    cleaned = _PATTERN.sub("", raw, count=1).lstrip()
    assert cleaned == "User text"
