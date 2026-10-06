"""Pre-summary efficiency regression tests for #23811 and #21470."""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


class TestCompressionSkipsWhenSummaryWouldInflate:
    """Regression: compression must not replace content with a larger summary.

    Issue #23811/#22037: the structured summary template has a _MIN_SUMMARY_TOKENS
    floor of 2000 plus ~500 tokens of overhead.  When the compressible window is
    tiny, the forced summary can be larger than the raw messages it replaces,
    causing rapid re-compression cycles.

    Guard (pre-summariser, #21470): bails BEFORE the summariser call when the
    approximate compressible window is not larger than the summary budget.  This
    saves an LLM call and prevents the inflation.

    Note: PR #25413 added a separate post-hoc guard (saved_estimate <= 0 after
    assembly) to catch verbose summaries the pre-guard missed.  We did NOT adopt
    it here because it conflicts with fallback-summary tests that expect
    compression to proceed on small conversations even when the fallback summary
    saves no tokens.  The pre-guard covers the common inflation case.
    """

    def _make_short_messages(self, n):
        """Return minimal alternating messages (very small token footprint)."""
        return [
            {"role": "user", "content": f"u{i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": f"a{i}"}
            for i in range(n)
        ]

    def test_pre_guard_bails_when_compressible_not_larger_than_budget(self):
        """Pre-guard: if compressible window is smaller than summary budget, skip it."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,  # 100K threshold
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
            _ = c.context_length

        # 8 alternating messages: head 2 + middle 3 + tail 3
        # Each message is ~4 chars -> ~1 token, so middle = ~3 tokens.
        # But the summary floor is 2000 tokens.  Compression should skip.
        # Set last_prompt_tokens above threshold to simulate a real session
        # where the guard fires based on auto-detected tokens.
        c.last_prompt_tokens = 150_000
        msgs = self._make_short_messages(8)
        original = msgs[:]
        result = c.compress(msgs, current_tokens=160_000)

        # Must return unchanged (guard short-circuits before any mutation)
        assert result == original
        # compression_count must NOT have been incremented
        assert c.compression_count == 0

    def test_pre_guard_proceeds_when_compressible_is_larger_than_budget(self):
        """Pre-guard: if compressible window is large enough, compression proceeds normally."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
            _ = c.context_length

        # Build 100 messages each large enough that the tail budget (10K)
        # can only hold a handful, leaving a substantial compressible middle.
        # With ~2000 tokens/message the tail fits ~5 messages -> middle = ~93
        # messages = ~186K tokens -> budget = max(2000, 186K*0.20) = 37,200.
        # Guard: 186000 > 37200 + 500  (compression proceeds)
        msgs = []
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"Message {i}: " + "x" * 8000})

        with patch.object(c, "_generate_summary", return_value="Mock summary"):
            result = c.compress(msgs, current_tokens=160_000)

        # Should have compressed (messages were dropped)
        assert len(result) < len(msgs)
        assert c.compression_count == 1
