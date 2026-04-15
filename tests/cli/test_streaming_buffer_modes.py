"""Tests for streaming buffer modes: character, word, and line.

Issue #9963: Improve terminal streaming output smoothness by supporting
character-level and word-level streaming instead of pure line-buffered.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from unittest.mock import patch, MagicMock
import shutil


def _make_cli_stub(buffer_mode="character"):
    """Create a minimal HermesCLI-like object with stream state."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_box_opened = True  # Pretend box is already open
    cli._stream_prefilt = ""
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._deferred_content = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False
    cli._stream_partial_line_open = False
    cli._streaming_buffer_mode = buffer_mode

    # Track all output
    cli._printed_lines = []     # _cprint calls (with newline)
    cli._printed_inline = []    # _cprint_inline calls (no newline)

    return cli


class TestCharacterMode:
    """Character mode should flush every token immediately."""

    def test_partial_text_flushed_immediately(self):
        """Text without newlines should appear immediately in character mode."""
        cli = _make_cli_stub("character")
        printed_inline = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline") as mock_inline:
            mock_inline.side_effect = lambda t: printed_inline.append(t)
            cli._emit_stream_text("Hello")
            # In character mode, "Hello" should be flushed inline immediately
            assert len(printed_inline) == 1
            assert "Hello" in printed_inline[0]

    def test_complete_line_uses_cprint(self):
        """Lines ending with \\n should use _cprint (with newline)."""
        cli = _make_cli_stub("character")
        printed_lines = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline"):
            mock_cprint.side_effect = lambda t: printed_lines.append(t)
            cli._emit_stream_text("Hello world\n")
            # Complete line should use _cprint
            assert len(printed_lines) == 1
            assert "Hello world" in printed_lines[0]

    def test_multi_token_sequence(self):
        """Multiple tokens should each appear immediately."""
        cli = _make_cli_stub("character")
        inline_calls = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline") as mock_inline:
            mock_inline.side_effect = lambda t: inline_calls.append(t)

            cli._emit_stream_text("The ")
            cli._emit_stream_text("quick ")
            cli._emit_stream_text("brown ")

            # Each token should produce an inline call
            assert len(inline_calls) == 3

    def test_newline_after_partial(self):
        """A newline arriving after partial content should complete the line."""
        cli = _make_cli_stub("character")
        cprint_calls = []
        inline_calls = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline") as mock_inline:
            mock_cprint.side_effect = lambda t: cprint_calls.append(t)
            mock_inline.side_effect = lambda t: inline_calls.append(t)

            cli._emit_stream_text("Hello")
            assert len(inline_calls) == 1  # Partial flushed inline
            assert cli._stream_partial_line_open is True

            cli._emit_stream_text(" world\n")
            # The newline should complete the line via _cprint
            assert len(cprint_calls) == 1
            assert "world" in cprint_calls[0]
            assert cli._stream_partial_line_open is False


class TestWordMode:
    """Word mode should flush at word/punctuation boundaries."""

    def test_partial_word_not_flushed(self):
        """Incomplete words should be kept in buffer."""
        cli = _make_cli_stub("word")
        inline_calls = []

        with patch("cli._cprint"), \
             patch("cli._cprint_inline") as mock_inline:
            mock_inline.side_effect = lambda t: inline_calls.append(t)

            cli._emit_stream_text("Hello")
            # No word boundary yet — should NOT flush
            assert len(inline_calls) == 0
            assert cli._stream_buf == "Hello"

    def test_word_boundary_triggers_flush(self):
        """Text with spaces should flush up to the last boundary."""
        cli = _make_cli_stub("word")
        inline_calls = []

        with patch("cli._cprint"), \
             patch("cli._cprint_inline") as mock_inline:
            mock_inline.side_effect = lambda t: inline_calls.append(t)

            cli._emit_stream_text("Hello world")
            # "Hello " should be flushed, "world" kept in buffer
            assert len(inline_calls) == 1
            assert "Hello " in inline_calls[0]
            assert cli._stream_buf == "world"

    def test_punctuation_boundary(self):
        """Punctuation like commas should also trigger flush."""
        cli = _make_cli_stub("word")
        inline_calls = []

        with patch("cli._cprint"), \
             patch("cli._cprint_inline") as mock_inline:
            mock_inline.side_effect = lambda t: inline_calls.append(t)

            cli._emit_stream_text("Hello, world")
            # "Hello, " should NOT be flushed because comma is at index 5
            # Actually "Hello," (up to index 5 inclusive) should be flushed
            assert len(inline_calls) == 1


class TestLineMode:
    """Line mode (legacy) should only emit on newlines."""

    def test_no_flush_without_newline(self):
        """Partial text should stay buffered in line mode."""
        cli = _make_cli_stub("line")
        inline_calls = []

        with patch("cli._cprint"), \
             patch("cli._cprint_inline") as mock_inline:
            mock_inline.side_effect = lambda t: inline_calls.append(t)

            cli._emit_stream_text("Hello world, this is a long paragraph")
            # Line mode: nothing should be flushed inline
            assert len(inline_calls) == 0
            assert "Hello world" in cli._stream_buf

    def test_newline_triggers_flush(self):
        """Newlines should emit complete lines even in line mode."""
        cli = _make_cli_stub("line")
        cprint_calls = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline"):
            mock_cprint.side_effect = lambda t: cprint_calls.append(t)

            cli._emit_stream_text("Hello\nWorld\n")
            assert len(cprint_calls) == 2


class TestFlushStream:
    """_flush_stream should handle partial line state correctly."""

    def test_flush_closes_partial_line(self):
        """Flushing with an open partial line should complete it."""
        cli = _make_cli_stub("character")
        cprint_calls = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline"):
            mock_cprint.side_effect = lambda t: cprint_calls.append(t)
            # Simulate state where partial line is open and buffer has content
            cli._stream_partial_line_open = True
            cli._stream_buf = "remaining text"
            cli._stream_box_opened = True

            with patch.object(shutil, "get_terminal_size",
                              return_value=os.terminal_size((80, 24))):
                cli._close_reasoning_box = lambda: None
                cli._flush_stream()

            # Should have flushed remaining text (without re-adding pad)
            # and then the box closing border
            flushed = [c for c in cprint_calls if "remaining text" in c]
            assert len(flushed) == 1
            assert cli._stream_partial_line_open is False

    def test_flush_empty_partial_line(self):
        """Flushing with open partial but empty buffer should close the line."""
        cli = _make_cli_stub("character")
        cprint_calls = []

        with patch("cli._cprint") as mock_cprint, \
             patch("cli._cprint_inline"):
            mock_cprint.side_effect = lambda t: cprint_calls.append(t)
            cli._stream_partial_line_open = True
            cli._stream_buf = ""
            cli._stream_box_opened = True

            with patch.object(shutil, "get_terminal_size",
                              return_value=os.terminal_size((80, 24))):
                cli._close_reasoning_box = lambda: None
                cli._flush_stream()

            # Should emit an empty _cprint to close the line
            assert cli._stream_partial_line_open is False


class TestResetState:
    """_reset_stream_state should clear all new state."""

    def test_reset_clears_partial_line(self):
        cli = _make_cli_stub("character")
        cli._stream_partial_line_open = True
        cli._reset_stream_state()
        assert cli._stream_partial_line_open is False

    def test_reset_clears_buffer(self):
        cli = _make_cli_stub("character")
        cli._stream_buf = "leftover"
        cli._reset_stream_state()
        assert cli._stream_buf == ""
