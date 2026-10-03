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


class TestCaseFoldKeepsIndices:
    """``str.lower()`` turns 'İ' (U+0130) into two code points, so a tag index found in the
    lowered copy drifts when it slices the original. Every streaming tag filter must give the
    same visible text however the stream is chunked — one delta or one char at a time."""

    TEXT = (
        "İyi.\n<memory-context>\nRECALLED\n</memory-context>\n"
        "İyi bir soru.\n<think>İİ HIDDEN plan</think>\nUçak en hızlısı.\n"
        "<think>İ HIDDEN tail"
    )

    @staticmethod
    def _think(deltas: list[str]) -> str:
        return _drive(StreamingThinkScrubber(), deltas)

    @staticmethod
    def _memory_context(deltas: list[str]) -> str:
        from agent.memory_manager import StreamingContextScrubber

        s = StreamingContextScrubber()
        return "".join(s.feed(d) for d in deltas) + s.flush()

    @staticmethod
    def _gateway(deltas: list[str]) -> str:
        from unittest.mock import MagicMock

        from gateway.stream_consumer import GatewayStreamConsumer

        c = GatewayStreamConsumer(MagicMock(), "chat")
        for d in deltas:
            c._filter_and_accumulate(d)
        c._flush_think_buffer()
        return c._accumulated

    @staticmethod
    def _cli(deltas: list[str]) -> str:
        from cli import HermesCLI

        c = HermesCLI.__new__(HermesCLI)
        c.show_reasoning = False
        c._stream_buf, c._stream_prefilt = "", ""
        c._stream_started = c._stream_box_opened = c._in_reasoning_block = False
        emitted: list[str] = []
        c._emit_stream_text = emitted.append
        c._stream_reasoning_delta = lambda _text: None
        for d in deltas:
            c._stream_delta(d)
        return "".join(emitted) + ("" if c._in_reasoning_block else c._stream_prefilt)

    @pytest.mark.parametrize("surface", ["_think", "_memory_context", "_gateway", "_cli"])
    def test_visible_text_is_independent_of_delta_boundaries(self, surface: str) -> None:
        run = getattr(self, surface)
        assert run([self.TEXT]) == run(list(self.TEXT))

    @pytest.mark.parametrize("surface", ["_think", "_gateway", "_cli"])
    def test_streamed_text_agrees_with_the_final_strip_on_one_code_point_case_maps(self, surface: str) -> None:
        """The Kelvin sign (U+212A) lowercases to ASCII 'k': the final-response strip hides a tag
        spelled with it, so the progressive stream must hide it too."""
        from agent.agent_runtime_helpers import strip_think_blocks

        text = "<THINK>HIDDEN</THINK>answer"
        assert getattr(self, surface)([text]) == strip_think_blocks(None, text)
