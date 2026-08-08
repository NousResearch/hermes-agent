"""Tests for ``_end_position`` — the whole-document replace range endpoint.

LSP Positions count lines by LSP line endings only (``\\n``, ``\\r\\n``,
``\\r``). ``str.splitlines()`` also breaks on ``\\v``, ``\\f``, the
information separators ``\\x1c``-``\\x1e``, ``\\x85`` (NEL) and the Unicode
separators U+2028 / U+2029, which would over-count lines and produce a wrong
end position for documents that contain those bytes.
"""

from __future__ import annotations

import pytest

from agent.lsp.client import _end_position


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", {"line": 0, "character": 0}),
        ("abc", {"line": 0, "character": 3}),
        ("abc\ndef", {"line": 1, "character": 3}),
        ("abc\n", {"line": 1, "character": 0}),
        ("abc\r\ndef", {"line": 1, "character": 3}),
        ("abc\r\n", {"line": 1, "character": 0}),
        ("abc\rdef", {"line": 1, "character": 3}),  # old-Mac CR is an LSP ending
        ("a\nb\nc", {"line": 2, "character": 1}),
    ],
)
def test_lsp_line_endings(text, expected):
    assert _end_position(text) == expected


@pytest.mark.parametrize(
    "sep",
    ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"],
)
def test_non_lsp_separators_do_not_start_a_new_line(sep):
    # These are NOT LSP line endings, so "a<sep>b" is a single 3-char line.
    assert _end_position(f"a{sep}b") == {"line": 0, "character": 3}
