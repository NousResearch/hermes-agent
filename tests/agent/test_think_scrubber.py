"""Tests for StreamingThinkScrubber.

These tests lock in the contract the scrubber must satisfy so downstream
consumers (ACP, api_server, TTS, CLI, gateway) never see reasoning
blocks leaking through the stream_delta_callback.  The scenarios map
directly to the MiniMax-M2.7 / DeepSeek / Qwen3 streaming patterns that
break the older per-delta regex strip.
"""

from __future__ import annotations

import pytest

from agent.think_scrubber import StreamingThinkScrubber


def _drive(scrubber: StreamingThinkScrubber, deltas: list[str]) -> str:
    """Feed a sequence of deltas and return the concatenated visible output."""
    out = [scrubber.feed(d) for d in deltas]
    out.append(scrubber.flush())
    return "".join(out)


class TestClosedPairs:
    """Closed <tag>...</tag> pairs are always stripped, regardless of boundary."""

    def test_closed_pair_single_delta(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>reasoning</think>Hello world"]) == "Hello world"


    @pytest.mark.parametrize(
        "tag",
        ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD"],
    )
    def test_all_tag_variants(self, tag: str) -> None:
        s = StreamingThinkScrubber()
        delta = f"<{tag}>x</{tag}>Hello"
        assert _drive(s, [delta]) == "Hello"



class TestUnterminatedOpen:
    """Unterminated open tag discards all subsequent content to end of stream."""

    def test_open_at_stream_start(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>reasoning text with no close"]) == ""



    def test_prose_mentioning_tag_not_stripped(self) -> None:
        """Mid-line '<think>' in prose is preserved (no boundary)."""
        s = StreamingThinkScrubber()
        text = "Use the <think> element for reasoning"
        assert _drive(s, [text]) == text


class TestOrphanClose:
    """Orphan close tags (no prior open) are stripped without boundary check."""

    def test_orphan_close_alone(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello</think>world"]) == "Helloworld"




class TestDsmlBlocks:
    """DSML tool-call markup (DeepSeek) must never reach stream callbacks (#115475).

    DSML is line-based: ``DSML | tool_calls`` … ``DSML | /tool_calls`` (optionally
    angle-bracketed ``<DSML | …>``). Value lines between ``parameter`` open/close are
    plain lines belonging to the block. Blocks may be split across deltas mid-line.
    """

    DSML_BLOCK = (
        'DSML | tool_calls\n'
        'DSML | invoke name="terminal"\n'
        'DSML | parameter name="command" string="true"\n'
        'python3 /path/to/script.py --flag\n'
        'DSML | /parameter\n'
        'DSML | /invoke\n'
        'DSML | /tool_calls\n'
    )

    def test_full_block_single_delta(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Intro\n" + self.DSML_BLOCK + "Outro"]) == "Intro\nOutro"

    def test_block_split_across_deltas(self) -> None:
        s = StreamingThinkScrubber()
        out = _drive(s, ["Intro\nDSML | tool", "_calls\nDSML | invoke name=\"t", "erminal\"\nrun --flag\nDSML | /tool_calls\nOutro"])
        assert "DSML" not in out
        assert "run --flag" not in out
        assert out == "Intro\nOutro"

    def test_angled_block_stripped(self) -> None:
        s = StreamingThinkScrubber()
        block = "<DSML | tool_calls>\n<DSML | invoke name=\"x\">\n<DSML | /invoke>\n<DSML | /tool_calls>\n"
        assert _drive(s, [block + "Visible."]) == "Visible."

    def test_function_results_block_stripped(self) -> None:
        s = StreamingThinkScrubber()
        block = 'DSML | function_results\nDSML | result_item name="terminal"\n{"ok":true}\nDSML | /result_item\nDSML | /function_results\n'
        assert _drive(s, [block + "Done."]) == "Done."

    def test_unterminated_block_discarded_at_flush(self) -> None:
        s = StreamingThinkScrubber()
        out = _drive(s, ["Working…\nDSML | tool_calls\nDSML | invoke name=\"terminal\"\nvalue line"])
        assert "DSML" not in out
        assert "value line" not in out
        # The boundary newline before an unterminated block goes with it, matching the
        # whole-string regex stripper's unterminated-block semantics.
        assert out == "Working…"

    def test_value_lines_only_inside_block_kept_outside(self) -> None:
        # Plain lines before/after the block are prose; identical text inside is dropped.
        s = StreamingThinkScrubber()
        out = _drive(s, ["before\nDSML | tool_calls\nmiddle\nDSML | /tool_calls\nafter"])
        assert out == "before\nafter"

    def test_prose_mentioning_dsml_mid_line_preserved(self) -> None:
        s = StreamingThinkScrubber()
        text = "The DSML | tool_calls syntax is interesting.\nReal answer."
        assert _drive(s, [text]) == text

    def test_held_partial_dsml_line_released_at_flush(self) -> None:
        # A trailing "DSML" that never completes into a directive is prose, not markup.
        s = StreamingThinkScrubber()
        assert _drive(s, ["Answer: DSML"]) == "Answer: DSML"


class TestPartialTagsAcrossDeltas:
    """Partial tags at delta boundaries must be held back, not emitted raw."""

    def test_split_open_tag_held_back(self) -> None:
        """'<' arrives alone, 'think>' completes it on next delta."""
        s = StreamingThinkScrubber()
        # At stream start, last_emitted_ended_newline=True, so <think> at 0 is boundary
        assert (
            _drive(s, ["<", "think>reasoning</think>done"])
            == "done"
        )

    def test_split_open_tag_not_at_boundary(self) -> None:
        """Mid-line split '<' + 'think>X</think>' is a closed pair.

        Closed pairs are always stripped (matching
        ``_strip_think_blocks`` case 1), even without a block
        boundary — a closed pair is an intentional bounded construct.
        """
        s = StreamingThinkScrubber()
        out = _drive(s, ["word<", "think>prose</think>more"])
        assert out == "wordmore"




class TestTheMiniMaxScenario:
    """The exact pattern run_agent per-delta regex strip breaks."""

    def test_minimax_split_open(self) -> None:
        """delta1='<think>', delta2='Let me check', delta3='</think>done'."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "Let me check their config", "</think>", "done"])
        assert out == "done"


    def test_minimax_unterminated_reasoning_at_end(self) -> None:
        """Unclosed reasoning at stream end is dropped entirely."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "The user wants", " to know something"])
        assert out == ""


class TestResetAndReentry:
    def test_reset_clears_in_block_state(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("<think>hanging")
        assert s._in_block is True
        s.reset()
        assert s._in_block is False
        # After reset, a new turn works cleanly
        assert _drive(s, ["Hello world"]) == "Hello world"

    def test_reset_clears_buffered_partial_tag(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("word<")
        assert s._buf == "<"
        s.reset()
        assert s._buf == ""
        assert _drive(s, ["fresh content"]) == "fresh content"


class TestFlushBehaviour:



    def test_flush_restores_stream_start_boundary(self) -> None:
        """End-of-stream flush must re-arm block-boundary gating.

        Thinking-only / empty-response retries flush then stream again
        without ``reset()``.  If flush left ``_last_emitted_ended_newline``
        False (e.g. after emitting a held-back ``<``), the next stream's
        opening ``<think>`` looked mid-line and leaked into the UI.
        """
        s = StreamingThinkScrubber()
        assert s.feed("word") == "word"
        assert s._last_emitted_ended_newline is False
        assert s.flush() == ""
        assert s._last_emitted_ended_newline is True
        assert (
            _drive(s, ["<think>", "secret reasoning", "</think>", "Visible answer"])
            == "Visible answer"
        )

    def test_flush_partial_tag_tail_does_not_poison_next_stream(self) -> None:
        """Flushing a held-back ``<`` must not make the next open tag leak."""
        s = StreamingThinkScrubber()
        s.feed("word<")
        assert s.flush() == "<"
        assert s._last_emitted_ended_newline is True
        assert _drive(s, ["<think>hidden</think>Hello"]) == "Hello"


class TestRealisticStreaming:
    """Character-by-character streaming must work as well as larger chunks."""

    def test_char_by_char_closed_pair(self) -> None:
        s = StreamingThinkScrubber()
        deltas = list("<think>x</think>Hello world")
        assert _drive(s, deltas) == "Hello world"


    def test_reasoning_then_real_response_first_word_preserved(self) -> None:
        """Regression: the first word of the final response must NOT be eaten.

        Stefan's screenshot bug — 'Let me check' was being rendered as
        ' me check'.  The scrubber must not consume any character of
        post-close content.
        """
        s = StreamingThinkScrubber()
        deltas = [
            "<think>",
            "User wants to know things",
            "</think>",
            "Let me check their config.",
        ]
        assert _drive(s, deltas) == "Let me check their config."

    def test_no_tag_passthrough_is_identical(self) -> None:
        """Streams without any reasoning tags pass through byte-for-byte."""
        s = StreamingThinkScrubber()
        deltas = ["Hello ", "world ", "how ", "are ", "you?"]
        assert _drive(s, deltas) == "Hello world how are you?"
