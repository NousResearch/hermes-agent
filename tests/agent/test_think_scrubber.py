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
        s.reset()
        # After reset, a new turn works cleanly
        assert _drive(s, ["Hello world"]) == "Hello world"

    def test_reset_clears_buffered_partial_tag(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("word<")
        s.reset()
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
        assert s.flush() == ""
        assert (
            _drive(s, ["<think>", "secret reasoning", "</think>", "Visible answer"])
            == "Visible answer"
        )

    def test_flush_partial_tag_tail_does_not_poison_next_stream(self) -> None:
        """Flushing a held-back ``<`` must not make the next open tag leak."""
        s = StreamingThinkScrubber()
        s.feed("word<")
        assert s.flush() == "<"
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


class TestChineseReasoningTags:
    """MiniMax-M3 emits Chinese reasoning tags (#43827); both surfaces must hide them."""

    def test_split_chinese_tag_scrubbed_in_stream(self) -> None:
        s = StreamingThinkScrubber()
        deltas = ["<思", "考>让我想想", "……</思考>", "\n答案是 42"]
        assert _drive(s, deltas) == "\n答案是 42"

    def test_final_response_strip_hides_chinese_tags(self) -> None:
        from agent.agent_runtime_helpers import strip_think_blocks

        out = strip_think_blocks(None, "<反思>内部推理</反思>最终答案\n<推理>未闭合的推理")
        assert "内部推理" not in out and "未闭合" not in out
        assert "最终答案" in out

    def test_cli_replay_strip_hides_chinese_tags(self) -> None:
        from cli import _strip_reasoning_tags

        assert _strip_reasoning_tags("<思考>secret</思考>答案") == "答案"


class TestMidLineOpenSplitClose:
    """A mid-line open tag whose CLOSE tag splits across deltas leaks the reasoning today
    (#128294): the pair only strips when complete within one buffer, and the boundary gate
    (prose-mention guard) never latches a mid-line open. A pending latch closes the hole
    without touching the mention semantics."""

    def test_midline_open_split_close_hidden(self) -> None:
        s = StreamingThinkScrubber()
        out = _drive(s, ["Let me check: <think>SECRET", " STUFF</think> ok"])
        assert "SECRET" not in out and "STUFF" not in out, out
        assert out == "Let me check:  ok"

    def test_midline_open_split_close_reasoning_in_last_hidden(self) -> None:
        s = StreamingThinkScrubber()
        _drive(s, ["Let me check: <think>SECRET", " STUFF</think> ok"])
        assert "SECRET" in s.last_hidden

    def test_thinking_variant_split_close_hidden(self) -> None:
        """A second known tag name rides the same latch (mm:think and other namespaces are
        #124761's face; this branch fixes the split-close mechanism for the known set)."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["hi <thinking>SECRET</think", "ing> ok"])
        assert "SECRET" not in out, out

    def test_mention_without_close_released_verbatim_at_flush(self) -> None:
        """The prose-mention case is preserved byte-for-byte: the pending latch holds the
        tail, and flush() (no close ever) re-releases the tag literal and the text."""
        text = "mentions <think> inline for reasoning"
        s = StreamingThinkScrubber()
        out = _drive(s, [text[:24], text[24:]])
        assert out == text

    def test_cap_exceeded_releases_and_streams_normally(self) -> None:
        """Beyond the retain cap the latch gives up and passes through (never worse than
        today's leak-tolerant behavior for prose); a later pair still strips via the pair
        branch."""
        import agent.think_scrubber as ts

        big = "x" * (ts.PENDING_RETAIN_CAP + 100)
        s = StreamingThinkScrubber()
        out = _drive(s, ["hi <think>" + big, " tail"])
        assert out == "hi <think>" + big + " tail"

    def test_boundary_open_keeps_hard_discard_semantics(self) -> None:
        """A boundary open still latches the hard block: unterminated reasoning at stream end
        is discarded, not released (flush docstring invariant)."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>SECRET", " more reasoning"])
        assert out == ""

    def test_pending_close_with_matching_name_only(self) -> None:
        """A close tag for a DIFFERENT known name must not confirm the pending block: it stays
        pending (and is released verbatim at flush if no matching close ever arrives)."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["x <think> a </thinking> b"])
        assert out == "x <think> a b"

    def test_reset_clears_pending(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("mid <think> held")
        s.reset()
        assert _drive(s, ["visible"]) == "visible"
