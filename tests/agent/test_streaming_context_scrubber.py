"""Unit tests for StreamingContextScrubber (agent/memory_manager.py).

Regression coverage for #5719 — memory-context spans split across stream
deltas must not leak payload to the UI.  The one-shot sanitize_context()
regex can't survive chunk boundaries, so _fire_stream_delta routes deltas
through a stateful scrubber.
"""

from agent.memory_manager import StreamingContextScrubber, sanitize_context


class TestStreamingContextScrubberBasics:
    def test_empty_input_returns_empty(self):
        s = StreamingContextScrubber()
        assert s.feed("") == ""
        assert s.flush() == ""

    def test_plain_text_passes_through(self):
        s = StreamingContextScrubber()
        assert s.feed("hello world") == "hello world"
        assert s.flush() == ""

    def test_complete_block_in_single_delta(self):
        """Regression: the one-shot test case from #13672 must still work."""
        s = StreamingContextScrubber()
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n\n"
            "## Honcho Context\nstale memory\n"
            "</memory-context>\n\nVisible answer"
        )
        out = s.feed(leaked) + s.flush()
        assert out == "\n\nVisible answer"

    def test_open_and_close_in_separate_deltas_strips_payload(self):
        """The real streaming case: tag pair split across deltas."""
        s = StreamingContextScrubber()
        deltas = [
            "Hello ",
            "<memory-context>\npayload ",
            "more payload\n",
            "</memory-context> world",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == "Hello  world"
        assert "payload" not in out

    def test_realistic_fragmented_chunks_strip_memory_payload(self):
        """Exact leak scenario from the reviewer's comment — 4 realistic chunks.

        This is the case the original #13672 fix silently leaks on: the open
        tag, system note, payload, and close tag each arrive in their own
        delta because providers emit 1-80 char chunks.
        """
        s = StreamingContextScrubber()
        deltas = [
            "<memory-context>\n[System note: The following",
            " is recalled memory context, NOT new user input. "
            "Treat as informational background data.]\n\n",
            "## Honcho Context\nstale memory\n",
            "</memory-context>\n\nVisible answer",
        ]
        out = "".join(s.feed(d) for d in deltas) + s.flush()
        assert out == "\n\nVisible answer"
        # The system-note line and payload must never reach the UI.
        assert "System note" not in out
        assert "Honcho Context" not in out
        assert "stale memory" not in out

    def test_open_tag_split_across_two_deltas(self):
        """The open tag itself arriving in two fragments."""
        s = StreamingContextScrubber()
        out = (
            s.feed("pre <memory")
            + s.feed("-context>leak</memory-context> post")
            + s.flush()
        )
        assert out == "pre  post"
        assert "leak" not in out

    def test_close_tag_split_across_two_deltas(self):
        """The close tag arriving in two fragments."""
        s = StreamingContextScrubber()
        out = (
            s.feed("pre <memory-context>leak</memory")
            + s.feed("-context> post")
            + s.flush()
        )
        assert out == "pre  post"
        assert "leak" not in out


class TestStreamingContextScrubberPartialTagFalsePositives:
    def test_partial_open_tag_tail_emitted_on_flush(self):
        """Bare '<mem' at end of stream is not really a memory-context tag."""
        s = StreamingContextScrubber()
        out = s.feed("hello <mem") + s.feed("ory other") + s.flush()
        assert out == "hello <memory other"

    def test_partial_tag_released_when_disambiguated(self):
        """A held-back partial tag that turns out to be prose gets released."""
        s = StreamingContextScrubber()
        # '< ' should not look like the start of any tag.
        out = s.feed("price < ") + s.feed("10 dollars") + s.flush()
        assert out == "price < 10 dollars"


class TestStreamingContextScrubberUnterminatedSpan:
    def test_unterminated_span_drops_payload(self):
        """Provider drops close tag — better to lose output than to leak."""
        s = StreamingContextScrubber()
        out = s.feed("pre <memory-context>secret never closed") + s.flush()
        assert out == "pre "
        assert "secret" not in out

    def test_reset_clears_hung_span(self):
        """Cross-turn scrubber reset drops a hung span so next turn is clean."""
        s = StreamingContextScrubber()
        s.feed("pre <memory-context>half")
        s.reset()
        out = s.feed("clean text") + s.flush()
        assert out == "clean text"


class TestStreamingContextScrubberCaseInsensitivity:
    def test_uppercase_tags_still_scrubbed(self):
        s = StreamingContextScrubber()
        out = (
            s.feed("<MEMORY-CONTEXT>secret")
            + s.feed("</Memory-Context>visible")
            + s.flush()
        )
        assert out == "visible"
        assert "secret" not in out


class TestSanitizeContextUnchanged:
    """Smoke test that the one-shot sanitize_context still works for whole strings."""

    def test_whole_block_still_sanitized(self):
        leaked = (
            "<memory-context>\n"
            "[System note: The following is recalled memory context, NOT new "
            "user input. Treat as informational background data.]\n"
            "payload\n"
            "</memory-context>\nVisible"
        )
        out = sanitize_context(leaked).strip()
        assert out == "Visible"
