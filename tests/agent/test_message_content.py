from __future__ import annotations

import logging
from types import SimpleNamespace

from agent.message_content import flatten_message_text


def test_flatten_message_text_accepts_chat_and_responses_text_parts():
    content = [
        {"type": "text", "text": "chat text"},
        {"type": "input_text", "text": "user text"},
        {"type": "output_text", "text": "assistant text"},
        {"type": "summary_text", "text": "summary text"},
    ]

    assert flatten_message_text(content) == "chat text\nuser text\nassistant text\nsummary text"


def test_flatten_message_text_accepts_object_parts():
    content = [
        SimpleNamespace(type="output_text", text="object text"),
        {"content": "legacy content"},
    ]

    assert flatten_message_text(content) == "object text\nlegacy content"


def test_flatten_message_text_logs_debug_when_str_fallback_raises(caplog):
    class PathologicalContent:
        def __str__(self):
            raise RuntimeError("c2 string conversion fixture")

        def __repr__(self):
            return "SECRET_USER_MESSAGE_CONTENT"

    with caplog.at_level(logging.DEBUG, logger="agent.message_content"):
        assert flatten_message_text(PathologicalContent()) == ""

    records = [
        record
        for record in caplog.records
        if record.name == "agent.message_content"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "PathologicalContent" in records[0].getMessage()
    assert "c2 string conversion fixture" in records[0].getMessage()
    assert "SECRET_USER_MESSAGE_CONTENT" not in records[0].getMessage()
