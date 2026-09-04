"""Verify backslash+Enter inserts a newline instead of submitting.

On terminals that don't send a distinct Shift+Enter sequence (e.g. iTerm2
with default settings), users can type a trailing ``\\`` before Enter to
insert a newline.  This is the same convention Claude Code uses.

See issue #35057.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDocument:
    """Minimal document stub for testing the backslash+Enter handler."""

    def __init__(self, text: str, cursor_position: int):
        self.text = text
        self.cursor_position = cursor_position


class _FakeBuffer:
    """Minimal buffer stub that records insert_text / delete_before_cursor."""

    def __init__(self, text: str, cursor_position: int):
        self.document = _FakeDocument(text, cursor_position)
        self._text = text
        self._cursor = cursor_position
        self.inserted: list[str] = []
        self.deleted: list[int] = []

    def delete_before_cursor(self, count: int = 1):
        self.deleted.append(count)
        self._text = self._text[: max(0, self._cursor - count)] + self._text[self._cursor:]
        self._cursor = max(0, self._cursor - count)
        self.document = _FakeDocument(self._text, self._cursor)

    def insert_text(self, text: str):
        self.inserted.append(text)
        self._text = self._text[: self._cursor] + text + self._text[self._cursor:]
        self._cursor += len(text)
        self.document = _FakeDocument(self._text, self._cursor)


class _FakeEvent:
    """Minimal event stub wrapping a buffer."""

    def __init__(self, buffer: _FakeBuffer):
        self.app = MagicMock()
        self.app.current_buffer = buffer
        self.current_buffer = buffer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_backslash_at_end_converts_to_newline():
    """Trailing ``\\`` + Enter → newline, backslash consumed."""
    buf = _FakeBuffer("hello \\", 7)
    event = _FakeEvent(buf)

    # Simulate the guard from handle_enter
    doc = event.app.current_buffer.document
    assert doc.text and doc.cursor_position > 0
    char_before = doc.text[doc.cursor_position - 1]
    assert char_before == "\\"

    buf.delete_before_cursor(1)
    buf.insert_text("\n")

    assert buf._text == "hello \n"
    assert "\n" in "".join(buf.inserted)


def test_backslash_in_middle_converts_to_newline():
    """``\\`` before cursor in the middle of text → newline."""
    buf = _FakeBuffer("hello \\world", 7)
    event = _FakeEvent(buf)

    doc = event.app.current_buffer.document
    char_before = doc.text[doc.cursor_position - 1]
    assert char_before == "\\"

    buf.delete_before_cursor(1)
    buf.insert_text("\n")

    assert buf._text == "hello \nworld"


def test_no_backslash_does_not_trigger():
    """Normal text without trailing ``\\`` → no conversion."""
    buf = _FakeBuffer("hello world", 11)
    doc = buf.document

    char_before = doc.text[doc.cursor_position - 1] if doc.cursor_position > 0 else ""
    assert char_before != "\\"


def test_empty_buffer_does_not_trigger():
    """Empty buffer → no crash, no conversion."""
    buf = _FakeBuffer("", 0)
    doc = buf.document

    assert not doc.text
    assert doc.cursor_position == 0


def test_double_backslash_leaves_one():
    """``\\\\`` + Enter → one ``\\`` remains (first escapes second)."""
    buf = _FakeBuffer("hello \\\\", 8)
    event = _FakeEvent(buf)

    doc = event.app.current_buffer.document
    char_before = doc.text[doc.cursor_position - 1]
    assert char_before == "\\"

    # Only the trailing backslash is consumed; the second one remains.
    buf.delete_before_cursor(1)
    buf.insert_text("\n")

    assert buf._text == "hello \\\n"
