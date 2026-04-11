"""Tests for _stream_delta's handling of <think> tags in prose vs real reasoning blocks."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest


def _make_cli_stub():
    """Create a minimal HermesCLI-like object with stream state."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli._stream_buf = ""
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._stream_prefilt = ""
    cli._in_reasoning_block = False
    cli._reasoning_stream_started = False
    cli._reasoning_box_opened = False
    cli._reasoning_buf = ""
    cli._reasoning_preview_buf = ""
    cli._deferred_content = ""
    cli._stream_text_ansi = ""
    cli._stream_needs_break = False
    cli._emitted = []

    # Mock _emit_stream_text to capture output
    def mock_emit(text):
        cli._emitted.append(text)
    cli._emit_stream_text = mock_emit

    # Mock _stream_reasoning_delta
    cli._reasoning_emitted = []
    def mock_reasoning(text):
        cli._reasoning_emitted.append(text)
    cli._stream_reasoning_delta = mock_reasoning

    return cli


class TestThinkTagInProse:
    """<think> mentioned in prose should NOT trigger reasoning suppression."""

    def test_think_tag_mid_sentence(self):
        """'(/think not producing <think> tags)' should pass through."""
        cli = _make_cli_stub()
        tokens = [
            "  1. Fix reasoning mode in eval ",
            "(/think not producing ",
            "<think>",
            " tags — ~2% gap)",
            "\n  2. Launch production",
        ]
        for t in tokens:
            cli._stream_delta(t)
        assert not cli._in_reasoning_block, "<think> in prose should not enter reasoning block"
        full = "".join(cli._emitted)
        assert "<think>" in full, "The literal <think> tag should be in the emitted text"
        assert "Launch production" in full

    def test_think_tag_after_text_on_same_line(self):
        """'some text <think>' should NOT trigger reasoning."""
        cli = _make_cli_stub()
        cli._stream_delta("Here is the <think> tag explanation")
        assert not cli._in_reasoning_block
        full = "".join(cli._emitted)
        assert "<think>" in full

    def test_think_tag_in_backticks(self):
        """'`<think>`' should NOT trigger reasoning."""
        cli = _make_cli_stub()
        cli._stream_delta("Use the `<think>` tag for reasoning")
        assert not cli._in_reasoning_block


class TestRealReasoningBlock:
    """Real <think> tags at block boundaries should still be caught."""

    def test_think_at_start_of_stream(self):
        """'<think>reasoning</think>answer' should suppress reasoning."""
        cli = _make_cli_stub()
        cli._stream_delta("<think>")
        assert cli._in_reasoning_block
        cli._stream_delta("I need to analyze this")
        cli._stream_delta("</think>")
        assert not cli._in_reasoning_block
        cli._stream_delta("Here is my answer")
        full = "".join(cli._emitted)
        assert "Here is my answer" in full
        assert "I need to analyze" not in full  # reasoning was suppressed

    def test_think_after_newline(self):
        """'text\\n<think>' should trigger reasoning block."""
        cli = _make_cli_stub()
        cli._stream_delta("Some preamble\n<think>")
        assert cli._in_reasoning_block
        full = "".join(cli._emitted)
        assert "Some preamble" in full

    def test_think_after_newline_with_whitespace(self):
        """'text\\n  <think>' should trigger reasoning block."""
        cli = _make_cli_stub()
        cli._stream_delta("Some preamble\n  <think>")
        assert cli._in_reasoning_block

    def test_think_with_only_whitespace_before(self):
        """'   <think>' (whitespace only prefix) should trigger."""
        cli = _make_cli_stub()
        cli._stream_delta("   <think>")
        assert cli._in_reasoning_block


class TestFlushRecovery:
    """_flush_stream should recover content from false-positive reasoning blocks."""

    def test_flush_recovers_buffered_content(self):
        """If somehow in reasoning block at flush, content is recovered."""
        cli = _make_cli_stub()
        # Manually set up a false-positive state
        cli._in_reasoning_block = True
        cli._stream_prefilt = " tags — ~2% gap)\n  2. Launch production"
        cli._stream_box_opened = True

        # Mock _close_reasoning_box and box closing
        cli._close_reasoning_box = lambda: None

        # Call flush
        from unittest.mock import patch
        import shutil
        with patch.object(shutil, "get_terminal_size", return_value=os.terminal_size((80, 24))):
            with patch("cli._cprint"):
                cli._flush_stream()

        assert not cli._in_reasoning_block
        full = "".join(cli._emitted)
        assert "Launch production" in full
