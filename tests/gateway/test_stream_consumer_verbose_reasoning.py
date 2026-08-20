"""Tests for verbose reasoning display in GatewayStreamConsumer.

Verifies that when ``verbose_reasoning=True``, think/thinking block content
from llama-server (and other content-stream providers) is buffered and emitted
with a ``💭 **Reasoning:** `` prefix instead of being silently dropped.

The reasoning content is appended directly into ``_accumulated`` at the inline
position where the closing tag was found, so it appears in the correct order:
text before → ``💭 **Reasoning:**`` content → text after.
"""

from unittest.mock import MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ── Helper functions ──────────────────────────────────────────────────────


def _make_consumer(verbose_reasoning: bool = False) -> GatewayStreamConsumer:
    """Create a test consumer with an optional ``verbose_reasoning`` flag."""
    cfg = StreamConsumerConfig(verbose_reasoning=verbose_reasoning)
    adapter = MagicMock()
    return GatewayStreamConsumer(adapter, "test_chat", config=cfg)


def _tag(open_name, close_name=None):
    """Construct a think tag pair from names (e.g. ('THINK', 'THINK'))."""
    lt = chr(60)  # <
    gt = chr(62)  # >
    close_name = close_name or open_name
    return f"{lt}{open_name}{gt}", f"{lt}/{close_name}{gt}"


PREFIX = "💭 **Reasoning:** "


# ── Baseline: verbose_reasoning=False still filters ───────────────────────


class TestFilteringWithoutVerbose:
    """Confirm that the default path still suppresses think blocks."""

    def test_think_block_suppressed_by_default(self):
        """Reasoning inside <think>...</think> is discarded when verbose=False."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=False)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Before{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate(f"Thinking...{NL}")
        consumer._filter_and_accumulate(close_tag)
        consumer._filter_and_accumulate("After")
        consumer._flush_think_buffer()

        assert consumer._accumulated == f"Before{NL}After"
        assert "Thinking..." not in consumer._accumulated

    def test_partial_tag_at_boundary(self):
        """Partial opening tag at buffer edge is held back, not lost.

        Simulates streaming chunks where the opening tag arrives piece by piece:
        first '<', then 'THIN', then 'K>'. The buffer should hold '<THIN' and
        only recognize the complete '<think>' tag once 'K>' arrives.
        """
        consumer = _make_consumer(verbose_reasoning=False)
        consumer._filter_and_accumulate("text")
        consumer._filter_and_accumulate("<")  # start of <think>
        consumer._filter_and_accumulate("THIN")  # partial: <THIN
        consumer._filter_and_accumulate("K>")  # completes: <think>
        consumer._filter_and_accumulate("hidden")
        consumer._filter_and_accumulate("</think>")
        consumer._filter_and_accumulate("visible")
        consumer._flush_think_buffer()

        assert consumer._accumulated == "textvisible"
        assert "hidden" not in consumer._accumulated

    def test_different_tag_variants_filtered(self):
        """All tag variants are suppressed when verbose=False."""
        NL = chr(10)
        for open_name, close_name in [
            ("THINK", "THINK"),
            ("REASONING_SCRATCHPAD", "REASONING_SCRATCHPAD"),
            ("THINKING", "THINKING"),
            ("thinking", "thinking"),
            ("thought", "thought"),
        ]:
            consumer = _make_consumer(verbose_reasoning=False)
            open_tag, close_tag = _tag(open_name, close_name)
            consumer._filter_and_accumulate(f"Before{NL}")
            consumer._filter_and_accumulate(open_tag)
            consumer._filter_and_accumulate("Thinking")
            consumer._filter_and_accumulate(close_tag)
            consumer._filter_and_accumulate("After")
            consumer._flush_think_buffer()
            assert "Thinking" not in consumer._accumulated, f"Failed for {open_name}"

    def test_think_tag_in_prose_not_triggered(self):
        """A think tag mentioned mid-sentence should NOT be treated as a block."""
        consumer = _make_consumer(verbose_reasoning=False)
        open_tag = _tag("THINK", "THINK")[0]
        consumer._filter_and_accumulate(
            "The model uses " + open_tag + " tags for reasoning, not "
            "markdown formatting."
        )
        consumer._flush_think_buffer()
        # The literal <think> tag should be in accumulated text (no match)
        assert "The model uses" in consumer._accumulated
        assert "markdown formatting." in consumer._accumulated


# ── Verbose: think block content is emitted with 💭 prefix ────────────────


class TestVerboseReasoningBasic:
    """Core behaviour: when verbose_reasoning=True, think content surfaces."""

    def test_single_think_block_emitted_with_prefix(self):
        """A complete <think>...</think> block emits 💭 **Reasoning:** content."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Before{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate(f"Thinking content{NL}")
        consumer._filter_and_accumulate(close_tag)
        consumer._filter_and_accumulate("After")
        consumer._flush_think_buffer()

        # Reasoning should be appended inline where the closing tag was found
        assert PREFIX in consumer._accumulated
        assert "Thinking content" in consumer._accumulated
        # Check ordering: text-before, reasoning, text-after
        idx_before = consumer._accumulated.index("Before")
        idx_reasoning = consumer._accumulated.index(PREFIX)
        idx_after = consumer._accumulated.index("After")
        assert idx_before < idx_reasoning < idx_after

    def test_text_before_and_after_think_block(self):
        """Non-think text accumulates normally around the think block."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Preamble{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate(f"Reasoning here{NL}")
        consumer._filter_and_accumulate(close_tag)
        consumer._filter_and_accumulate("Conclusion")
        consumer._flush_think_buffer()

        assert "Preamble" in consumer._accumulated
        assert "Conclusion" in consumer._accumulated
        assert PREFIX in consumer._accumulated
        assert "Reasoning here" in consumer._accumulated

    def test_multiple_think_blocks(self):
        """Multiple think blocks each emit their own 💭 **Reasoning:** block."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Start{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate(f"First thought{NL}")
        consumer._filter_and_accumulate(close_tag)
        consumer._filter_and_accumulate(f"Middle{NL}")
        consumer._filter_and_accumulate(close_tag)  # Orphan close (no match)
        consumer._filter_and_accumulate("End")
        consumer._flush_think_buffer()

        assert "Start" in consumer._accumulated
        assert "Middle" in consumer._accumulated
        assert "End" in consumer._accumulated
        # Only one thinking block was closed (the first one)
        assert consumer._accumulated.count(PREFIX) == 1
        assert "First thought" in consumer._accumulated


class TestVerboseReasoningEdgeCases:
    """Edge cases: stream ends mid-think, partial tags, case variants."""

    def test_stream_ends_inside_think_block(self):
        """If the stream ends while inside a think block, flush emits the buffer."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Before{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Incomplete thinking")
        consumer._flush_think_buffer()

        assert "Before" in consumer._accumulated
        assert PREFIX in consumer._accumulated
        assert "Incomplete thinking" in consumer._accumulated
        # Verify prefix appears before the content
        assert consumer._accumulated.index(PREFIX) < consumer._accumulated.index(
            "Incomplete thinking"
        )

    def test_partial_closing_tag_at_buffer_edge(self):
        """Partial closing tag at buffer edge is held in _think_buffer, not emitted yet."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Thinking content")
        consumer._filter_and_accumulate("</think")  # partial close
        consumer._filter_and_accumulate("ing>")     # completes the tag
        consumer._flush_think_buffer()

        assert PREFIX in consumer._accumulated
        assert "Thinking content" in consumer._accumulated

    def test_think_tag_in_prose_not_triggered_verbose(self):
        """A think tag mentioned mid-sentence should NOT be treated as a block (verbose=True)."""
        consumer = _make_consumer(verbose_reasoning=True)
        open_tag = _tag("THINK", "THINK")[0]
        consumer._filter_and_accumulate(
            "The model uses " + open_tag + " tags for reasoning, not "
            "markdown formatting."
        )
        consumer._flush_think_buffer()

        # The literal think tag should be in accumulated text (no match)
        assert "The model uses" in consumer._accumulated
        assert "markdown formatting." in consumer._accumulated
        assert PREFIX not in consumer._accumulated

    def test_orphan_close_tag_stripped(self):
        """An orphan closing tag (no matching open) is stripped from output."""
        consumer = _make_consumer(verbose_reasoning=True)
        close_tag = _tag("THINK", "THINK")[1]
        consumer._filter_and_accumulate(f"Before{close_tag}After")
        consumer._flush_think_buffer()

        assert "Before" in consumer._accumulated
        assert "After" in consumer._accumulated
        assert close_tag not in consumer._accumulated
        assert PREFIX not in consumer._accumulated


class TestVerboseReasoningBuffering:
    """Buffering: content accumulates across chunks before being flushed."""

    def test_reasoning_content_accumulates_across_chunks(self):
        """Multiple chunks inside a single think block accumulate into one emit."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Line 1 ")
        consumer._filter_and_accumulate("Line 2 ")
        consumer._filter_and_accumulate("Line 3")
        consumer._filter_and_accumulate(close_tag)
        consumer._flush_think_buffer()

        assert PREFIX in consumer._accumulated
        assert "Line 1" in consumer._accumulated
        assert "Line 2" in consumer._accumulated
        assert "Line 3" in consumer._accumulated
        # Single prefix for the entire block
        assert consumer._accumulated.count(PREFIX) == 1

    def test_reasoning_prefix_added_only_once(self):
        """The 💭 **Reasoning:** prefix should appear only once per block."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Reasoning ")
        consumer._filter_and_accumulate("content")
        consumer._filter_and_accumulate(close_tag)
        consumer._flush_think_buffer()

        assert consumer._accumulated.count(PREFIX) == 1
        assert "Reasoning content" in consumer._accumulated


class TestResetSegmentState:
    """Verify that segment breaks reset reasoning state."""

    def test_reset_segment_clears_reasoning_state(self):
        """After a segment break, reasoning buffers should be cleared."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Before{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Thinking content")

        assert consumer._reasoning_content == "Thinking content"
        assert not consumer._reasoning_prefix_added

        consumer._reset_segment_state()

        assert consumer._reasoning_content == ""
        assert not consumer._reasoning_prefix_added
        # Segment break also clears _accumulated, so reasoning content is lost
        assert consumer._accumulated == ""

    def test_flush_think_buffer_emits_unfinished_reasoning(self):
        """When stream ends mid-think, _flush_think_buffer emits the buffer."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=True)
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Incomplete thinking")

        consumer._flush_think_buffer()

        assert PREFIX in consumer._accumulated
        assert "Incomplete thinking" in consumer._accumulated


class TestVerboseReasoningDisabled:
    """Verify that when verbose_reasoning=False, everything is filtered."""

    def test_no_reasoning_when_disabled(self):
        """With verbose_reasoning=False, no reasoning content should appear."""
        open_tag, close_tag = _tag("THINK", "THINK")
        consumer = _make_consumer(verbose_reasoning=False)
        NL = chr(10)
        consumer._filter_and_accumulate(f"Before{NL}")
        consumer._filter_and_accumulate(open_tag)
        consumer._filter_and_accumulate("Thinking")
        consumer._filter_and_accumulate(close_tag)
        consumer._filter_and_accumulate("After")
        consumer._flush_think_buffer()

        assert "Thinking" not in consumer._accumulated
        assert PREFIX not in consumer._accumulated
        assert consumer._accumulated == f"Before{NL}After"
