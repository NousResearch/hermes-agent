"""Unit tests for the stream-json NDJSON parser."""

import pytest

from agent.claude_cli.errors import ProtocolError
from agent.claude_cli.protocol import StreamJsonParser


def test_single_complete_line_yields_one_event():
    """Feeding one complete JSON line produces exactly one event."""
    parser = StreamJsonParser()
    events = list(parser.feed(b'{"type": "assistant", "text": "hello"}\n'))
    assert events == [{"type": "assistant", "text": "hello"}]


def test_multiple_lines_in_one_chunk_yield_multiple_events():
    """Multiple newline-separated JSON objects in one chunk yield ordered events."""
    parser = StreamJsonParser()
    chunk = (
        b'{"type": "system", "session_id": "abc"}\n'
        b'{"type": "assistant", "text": "hi"}\n'
        b'{"type": "result", "exit_code": 0}\n'
    )
    events = list(parser.feed(chunk))
    assert events == [
        {"type": "system", "session_id": "abc"},
        {"type": "assistant", "text": "hi"},
        {"type": "result", "exit_code": 0},
    ]


def test_close_with_no_pending_data_yields_nothing():
    """Closing a parser that's been fully drained yields no extra events."""
    parser = StreamJsonParser()
    list(parser.feed(b'{"type": "assistant"}\n'))
    assert list(parser.close()) == []


def test_empty_lines_are_skipped():
    """Blank lines (e.g., from CRLF or padding) are tolerated, not parsed."""
    parser = StreamJsonParser()
    events = list(parser.feed(b'\n{"type": "assistant"}\n\n\n'))
    assert events == [{"type": "assistant"}]
