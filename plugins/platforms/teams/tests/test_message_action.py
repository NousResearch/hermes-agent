"""Tests for the 'Ask Hermes about this' message-action extraction."""

from __future__ import annotations

from plugins.platforms.teams.adapter import TeamsAdapter

extract = TeamsAdapter._extract_message_action


def test_extract_html_body_and_command():
    value = {
        "commandId": "askHermes",
        "messagePayload": {"body": {"content": "<div>What is <b>this</b>?</div>"}},
    }
    cmd, quoted = extract(value)
    assert cmd == "askHermes"
    assert quoted == "What is this ?".replace("  ", " ") or "this" in quoted
    assert "<" not in quoted


def test_extract_plain_text_payload():
    cmd, quoted = extract({"command_id": "askHermes", "messagePayload": {"text": "hello world"}})
    assert cmd == "askHermes" and quoted == "hello world"


def test_extract_object_shaped_value():
    class V:
        def __init__(self):
            self.commandId = "askHermes"
            self.messagePayload = {"text": "from object"}

    cmd, quoted = extract(V())
    assert cmd == "askHermes" and quoted == "from object"


def test_extract_empty_when_no_text():
    cmd, quoted = extract({"commandId": "askHermes", "messagePayload": {}})
    assert cmd == "askHermes" and quoted == ""


def test_extract_handles_none():
    cmd, quoted = extract(None)
    assert cmd == "" and quoted == ""
