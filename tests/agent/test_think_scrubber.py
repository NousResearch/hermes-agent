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

    def test_closed_pair_surrounded_by_content(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello <think>note</think> world"]) == "Hello  world"

    @pytest.mark.parametrize(
        "tag",
        ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD"],
    )
    def test_all_tag_variants(self, tag: str) -> None:
        s = StreamingThinkScrubber()
        delta = f"<{tag}>x</{tag}>Hello"
        assert _drive(s, [delta]) == "Hello"

    def test_case_insensitive_pair(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<THINK>x</Think>Hello"]) == "Hello"


class TestUnterminatedOpen:
    """Unterminated open tag discards all subsequent content to end of stream."""

    def test_open_at_stream_start(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>reasoning text with no close"]) == ""

    def test_open_after_newline(self) -> None:
        s = StreamingThinkScrubber()
        # 'Hello\n' is a block boundary for the <think> that follows
        assert _drive(s, ["Hello\n<think>reasoning"]) == "Hello\n"

    def test_open_after_newline_then_whitespace(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello\n  <think>reasoning"]) == "Hello\n  "

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

    def test_orphan_close_with_trailing_space_consumed(self) -> None:
        """Matches _strip_think_blocks case 3 \\s* behaviour."""
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello</think> world"]) == "Helloworld"

    def test_multiple_orphan_closes(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["A</think>B</thinking>C"]) == "ABC"


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

    def test_split_close_tag_held_back(self) -> None:
        """Close tag split across deltas still closes the block."""
        s = StreamingThinkScrubber()
        assert (
            _drive(s, ["<think>reasoning<", "/think>after"])
            == "after"
        )

    def test_split_close_tag_deep(self) -> None:
        """Close tag can be split anywhere."""
        s = StreamingThinkScrubber()
        assert (
            _drive(s, ["<think>reasoning</th", "ink>after"])
            == "after"
        )


class TestTheMiniMaxScenario:
    """The exact pattern run_agent per-delta regex strip breaks."""

    def test_minimax_split_open(self) -> None:
        """delta1='<think>', delta2='Let me check', delta3='</think>done'."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "Let me check their config", "</think>", "done"])
        assert out == "done"

    def test_minimax_split_open_with_trailing_content(self) -> None:
        """Reasoning then closes and hands off to final content."""
        s = StreamingThinkScrubber()
        out = _drive(
            s,
            [
                "<think>",
                "The user wants to know if thinking is on",
                "</think>",
                "\n\nshow_reasoning: false — thinking is OFF.",
            ],
        )
        assert out == "\n\nshow_reasoning: false — thinking is OFF."

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
    def test_flush_drops_unterminated_block(self) -> None:
        s = StreamingThinkScrubber()
        assert s.feed("<think>reasoning with no close") == ""
        assert s.flush() == ""

    def test_flush_emits_innocent_partial_tag_tail(self) -> None:
        """If held-back tail turned out not to be a real tag, emit it."""
        s = StreamingThinkScrubber()
        s.feed("word<")  # '<' could be a tag prefix
        # Stream ends with only '<' held back — emit it as prose.
        assert s.flush() == "<"

    def test_flush_on_empty_scrubber(self) -> None:
        s = StreamingThinkScrubber()
        assert s.flush() == ""

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

    def test_char_by_char_orphan_close(self) -> None:
        s = StreamingThinkScrubber()
        deltas = list("Hello</think>world")
        assert _drive(s, deltas) == "Helloworld"

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

class TestGemmaChannelThought:
    """Gemma 4's asymmetric channel-thought pair: <|channel>thought … <channel|>.

    Gemma 4 embeds reasoning in content using an asymmetric tag pair
    (model card, "Thinking Mode Configuration"):

        <|channel>thought\n[Internal reasoning]<channel|>Final answer

    Unlike Qwen's <think>…</think>, the close tag does not mirror the
    open tag and does not start with "</".  These tests lock in the
    suppression contract so Gemma reasoning never leaks to consumers.
    """

    def test_closed_pair_single_delta(self) -> None:
        s = StreamingThinkScrubber()
        assert (
            _drive(s, ["<|channel>thought\nsecret reasoning<channel|>Hello world"])
            == "Hello world"
        )

    def test_closed_pair_split_across_deltas(self) -> None:
        s = StreamingThinkScrubber()
        deltas = [
            "<|channel>thought\n",
            "The user wants me to write a Hello World program in Go. ",
            "I have already written the file hello.go using write_file.",
            "<channel|>",
            "Go言語は現在インストールされていないようです。",
        ]
        assert _drive(s, deltas) == "Go言語は現在インストールされていないようです。"

    def test_open_tag_split_mid_tag(self) -> None:
        s = StreamingThinkScrubber()
        deltas = ["<|chan", "nel>thou", "ght\nhidden", "<chan", "nel|>Visible"]
        assert _drive(s, deltas) == "Visible"

    def test_char_by_char(self) -> None:
        s = StreamingThinkScrubber()
        deltas = list("<|channel>thought\nx<channel|>Hello")
        assert _drive(s, deltas) == "Hello"

    def test_empty_thought_block_thinking_disabled(self) -> None:
        """Thinking disabled: Gemma still emits the tags with an empty body."""
        s = StreamingThinkScrubber()
        assert (
            _drive(s, ["<|channel>thought\n<channel|>Final answer"])
            == "Final answer"
        )

    def test_unterminated_open_discards_to_stream_end(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<|channel>thought\nreasoning with no close"]) == ""

    def test_orphan_close_tag_stripped(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello <channel|>world"]) == "Hello world"

    def test_prose_mention_mid_line_not_suppressed(self) -> None:
        """A mid-line prose mention of the open tag is not a block boundary."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["Gemma uses <|channel>thought to open reasoning."])
        assert out.startswith("Gemma uses ")

    def test_symmetric_tags_still_work_alongside(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>a</think>Hi"]) == "Hi"
        s2 = StreamingThinkScrubber()
        assert _drive(s2, ["<|channel>thought\nb<channel|>Hi"]) == "Hi"

    def test_glued_open_tag_no_newline(self) -> None:
        """Real Gemma output glues the tag to the reasoning: no newline."""
        s = StreamingThinkScrubber()
        deltas = [
            "<|channel>thoughtThe user wants me to ",
            "write Hello World in Go.<channel|>",
            "Answer text",
        ]
        assert _drive(s, deltas) == "Answer text"


class TestReasoningSink:
    """Suppressed block content must be routed to the on_reasoning sink.

    The sink is wired to the agent's reasoning delta callback so GUIs
    (desktop, TUI) render inline reasoning in the collapsible Thinking
    section — matching the display providers get when they return
    structured reasoning_content.  Content deltas must stay unchanged.
    """

    def _make(self):
        chunks: list[str] = []
        s = StreamingThinkScrubber(on_reasoning=chunks.append)
        return s, chunks

    def test_closed_pair_routes_inner_content(self) -> None:
        s, chunks = self._make()
        assert _drive(s, ["<think>secret plan</think>Hello"]) == "Hello"
        assert "".join(chunks) == "secret plan"

    def test_gemma_channel_pair_routes_inner_content(self) -> None:
        s, chunks = self._make()
        deltas = [
            "<|channel>thoughtThe user wants Go hello world. ",
            "I'll check the toolchain.<channel|>",
            "Go言語は現在インストールされていないようです。",
        ]
        assert _drive(s, deltas) == "Go言語は現在インストールされていないようです。"
        assert "".join(chunks) == (
            "The user wants Go hello world. I'll check the toolchain."
        )

    def test_streamed_block_routes_progressively(self) -> None:
        s, chunks = self._make()
        out = [s.feed("<think>"), s.feed("part one "), s.feed("part two")]
        # Reasoning should flow to the sink DURING the stream, not only
        # after the close tag arrives.
        assert "part one" in "".join(chunks)
        out.append(s.feed("</think>Visible"))
        out.append(s.flush())
        assert "".join(out) == "Visible"
        assert "".join(chunks) == "part one part two"

    def test_unterminated_block_flush_routes_tail(self) -> None:
        s, chunks = self._make()
        assert _drive(s, ["<think>never closed reasoning"]) == ""
        assert "".join(chunks) == "never closed reasoning"

    def test_no_block_no_reasoning_emitted(self) -> None:
        s, chunks = self._make()
        assert _drive(s, ["Plain visible text"]) == "Plain visible text"
        assert chunks == []

    def test_sink_error_does_not_break_content_stream(self) -> None:
        def _boom(_text: str) -> None:
            raise RuntimeError("sink died")

        s = StreamingThinkScrubber(on_reasoning=_boom)
        assert _drive(s, ["<think>x</think>Hello world"]) == "Hello world"

    def test_default_no_sink_still_suppresses(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>x</think>Hi"]) == "Hi"
