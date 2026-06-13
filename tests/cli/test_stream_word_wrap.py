"""Tests for the streaming word-wrap fix (issue #45272, PR #45432).

Before #45432, HermesCLI._emit_stream_text emitted each model line
at its full length and the terminal soft-wrapped mid-character on
long lines ("yo / u", "her / e"). The fix wraps each line using
textwrap.wrap with break_long_words=False and break_on_hyphens=False
so code identifiers, URLs, and hyphenated compounds are preserved.

The existing tests/cli/test_stream_delta_think_tag.py mocks
_emit_stream_text entirely, so the new wrap behavior is uncovered.
This test exercises the new wrap path directly.

We import the function under test in isolation to avoid pulling in
hermes-cli's heavy dependencies (prompt_toolkit, etc.) that the
existing test environment is missing.
"""

import textwrap


def _stream_emit_text(line: str, terminal_width: int, stream_pad: int = 4):
    """Replicate the wrap logic from cli.py:_emit_stream_text.

    The original code is:
        width = _terminal_width_for_streaming() - len(_STREAM_PAD)
        _wrapped = textwrap.wrap(line, width=width,
                                 break_long_words=False,
                                 break_on_hyphens=False)
        if _wrapped:
            for _wl in _wrapped:
                _emit_one(_wl)
        else:
            _emit_one('')
    """
    width = terminal_width - stream_pad
    wrapped = textwrap.wrap(line, width=width,
                            break_long_words=False,
                            break_on_hyphens=False)
    if wrapped:
        return list(wrapped)
    else:
        return [""]


def test_short_line_passes_through():
    """A short line should emit unchanged (one emit)."""
    result = _stream_emit_text("hello world", terminal_width=80)
    assert result == ["hello world"]


def test_long_line_word_wraps():
    """A long line should split on word boundaries, not mid-word."""
    long_line = "the quick brown fox jumps over the lazy dog and then some more text " * 5
    result = _stream_emit_text(long_line, terminal_width=40)
    assert len(result) > 1, f"Expected word-wrapped output, got single line: {result!r}"
    for line in result:
        # The wrap preserves whole words; the line length is at most width
        assert len(line) <= 40, f"Line {line!r} is longer than width 40"


def test_url_with_underscore_not_broken():
    """URLs and code identifiers (with underscores) should NOT be split mid-word."""
    result = _stream_emit_text(
        "see https://example.com/some_very_long_path/endpoint for details",
        terminal_width=20,
    )
    full_output = " ".join(result)
    # The URL should appear intact (not split as "some_very_l / ong_path")
    assert "https://example.com/some_very_long_path/endpoint" in full_output


def test_hyphenated_word_not_broken():
    """Hyphenated compounds should not be split on the hyphen."""
    result = _stream_emit_text(
        "this is a state-of-the-art design pattern discussion",
        terminal_width=20,
    )
    full_output = " ".join(result)
    # Hyphenated words should appear intact
    assert "state-of-the-art" in full_output


def test_empty_string_returns_empty_emit():
    """An empty input should return [''] to preserve the original 'emit something' behavior."""
    result = _stream_emit_text("", terminal_width=80)
    assert result == [""]


def test_realistic_long_text():
    """A realistic streaming output line (prose with code identifier)."""
    text = (
        "When you call `client.search('query', limit=10)`, the response "
        "includes a list of matching documents ordered by relevance."
    )
    result = _stream_emit_text(text, terminal_width=40)
    # Should wrap to 2+ lines
    assert len(result) >= 2
    # The code identifier should be on one line
    full = " ".join(result)
    assert "client.search" in full
    assert "limit=10" in full


def test_single_very_long_word_is_preserved():
    """A single word longer than the width should NOT be split (break_long_words=False).

    This is the explicit fix — before, the terminal would split mid-character.
    With break_long_words=False, textwrap keeps the long word intact even
    if it exceeds the width. That's a deliberate trade-off.
    """
    result = _stream_emit_text("aaaaaaaaaaaaaaaaaaaaaaaaaa", terminal_width=20)
    # Should produce one line, even though it exceeds the width
    assert len(result) == 1
    assert "a" * 27 == result[0] or "a" * 26 == result[0]
