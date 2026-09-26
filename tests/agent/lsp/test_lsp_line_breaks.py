"""Regression tests: LSP line counting must use LSP line breaks only (\\n, \\r\\n, \\r), not
Python's ``str.splitlines()``, which also splits on form feed, vertical tab, and other unicode
separators (\\x0b \\x0c \\x1c \\x1d \\x1e \\x85 \\u2028 \\u2029).

``agent/lsp/client.py::_end_position`` builds the end-of-document position for a whole-document
``textDocument/didChange`` (sync kind 2). Reporting a line past the document's real last line
makes strict servers reject the edit as an invalid range.

``agent/lsp/range_shift.py::build_line_shift`` maps pre-edit diagnostic line numbers to post-edit
line numbers for the diagnostics delta filter; miscounting lines shifts every diagnostic below a
form-feed (or similar) onto the wrong line.
"""

from __future__ import annotations

import re

from agent.lsp.client import _end_position, _lsp_splitlines
from agent.lsp.range_shift import build_line_shift

_LSP_LINE_BREAK = re.compile(r"\r\n|\r|\n")


class TestEndPosition:
    def test_matches_lsp_line_count_for_ordinary_text(self):
        for text in ("", "abc", "a\nb\nc", "a\nb\nc\n", "a\r\nb\r\nc", "a\rb\rc\r"):
            expected_lines = _LSP_LINE_BREAK.split(text) if text else [""]
            assert _end_position(text)["line"] == len(expected_lines) - 1

    def test_form_feed_is_not_treated_as_a_line_break(self):
        text = "x = 1\n\x0c\ny = 2"
        # LSP-correct: 3 lines (indices 0, 1, 2); str.splitlines() would see 4.
        assert _end_position(text) == {"line": 2, "character": 5}

    def test_unicode_line_separators_are_not_treated_as_line_breaks(self):
        for char in ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " "):
            text = f"a{char}b"
            # A single logical line: str.splitlines() would wrongly report 2.
            assert _end_position(text) == {"line": 0, "character": len(text)}

    def test_trailing_newline_still_reports_the_extra_empty_line(self):
        assert _end_position("a\nb\n") == {"line": 2, "character": 0}

    def test_utf16_character_offset_preserved(self):
        # Regression guard for d23d6e82 (UTF-16 units) staying intact alongside this fix.
        text = "a\n\U0001F600"  # astral character = 2 UTF-16 code units
        assert _end_position(text) == {"line": 1, "character": 2}


class TestLspSplitlines:
    def test_matches_str_splitlines_for_ordinary_text(self):
        for text in ("", "a", "a\nb", "a\nb\n", "a\n\nb", "a\n\n", "\n", "a\r\nb\r\n"):
            assert _lsp_splitlines(text) == text.splitlines()

    def test_does_not_split_on_form_feed(self):
        assert _lsp_splitlines("a\x0cb\nc") == ["a\x0cb", "c"]


class TestBuildLineShift:
    def test_form_feed_line_is_not_miscounted(self):
        # 'a\x0cb' is ONE LSP line (index 0); 'c' is line 1, 'd' is line 2.
        pre = "a\x0cb\nc\nd\n"
        post = "a\x0cb\nX\nc\nd\n"  # a new line inserted before 'c'
        shift = build_line_shift(pre, post)
        assert shift(1) == 2  # 'c' moves from pre-line 1 to post-line 2

    def test_identical_text_is_identity(self):
        text = "a\x0cb\nc\nd\n"
        shift = build_line_shift(text, text)
        assert shift(0) == 0
        assert shift(1) == 1
