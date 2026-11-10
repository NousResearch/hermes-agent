"""Transcript values retain the stored content serialization rules."""

from datetime import datetime, timezone, UTC

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession


@pytest.mark.parametrize("content, expected", [("bad\ud800", "bad\ufffd"), ("\x00json:[1, 2]", "\x00json:[1, 2]"), ("\x00json:not json", "\x00json:not json"), ((1, {"text": "tuple"}), [1, {"text": "tuple"}]), (12, "12"), (1.25, "1.25")])
def test_transcript_content_roundtrips_through_stored_form(backend, scope_a, content, expected):
    scope = scope_a
    backend.open(scope).sessions.create(NewSession("content", "cli"))
    backend.open(scope).messages.append("content", NewMessage("user", content, timestamp=datetime.fromtimestamp(1, UTC)))
    rows = backend.open(scope).messages.read_transcript("content")
    assert len(rows) == 1
    assert rows[0].content == expected
