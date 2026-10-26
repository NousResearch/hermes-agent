"""Regression test for #75588 — short tool-only suffix can make context
compressor scan past messages, causing IndexError in _find_context_summaries()."""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        return c


class TestTailCutBoundaryClamp:
    """Verify that _find_tail_cut_by_tokens never returns > len(messages).

    When a short conversation ends in a tool-call/result group and the
    protected head alignment reaches the end of the list, the tail-cut
    function used to return len(messages) + 1, which then caused
    _find_context_summaries() to index past the array boundary.  (#75588)
    """

    def _make_tool_group(self, call_id, n_results=1):
        msgs = [{"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "x", "arguments": "{}"}}]}]
        for i in range(n_results):
            msgs.append({"role": "tool", "content": f"result {i}", "tool_call_id": call_id})
        return msgs

    def test_tail_cut_never_exceeds_len_messages(self, compressor):
        """Simulate the exact bounds from the issue: head_end reaches n,
        so max(cut_idx, head_end+1) would produce n+1."""
        # Build a short transcript ending in a tool group
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            *self._make_tool_group("tc1", n_results=2),
        ]
        n = len(messages)
        # Force head_end to cover everything up to n (the protected head
        # swallowing the entire message list)
        head_end = n
        result = compressor._find_tail_cut_by_tokens(messages, head_end)
        assert result <= n, (
            f"_find_tail_cut_by_tokens returned {result} for len(messages)={n}; "
            "it must never exceed len(messages)"
        )

    def test_tail_cut_with_head_at_last_message(self, compressor):
        """head_end = n-1 (last message is the only unprotected one)."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]
        n = len(messages)
        result = compressor._find_tail_cut_by_tokens(messages, n - 1)
        assert result <= n

    def test_tail_cut_with_empty_tail(self, compressor):
        """head_end = n (no messages available for the tail at all)."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ]
        n = len(messages)
        result = compressor._find_tail_cut_by_tokens(messages, n)
        assert result <= n


class TestFindContextSummariesDefensiveClamp:
    """Verify that _find_context_summaries clamps its start/end bounds
    defensively, so it never raises IndexError even with bad caller input."""

    def test_out_of_range_end_does_not_crash(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        # end > len(messages) should not crash
        result = ContextCompressor._find_context_summaries(messages, 0, 999)
        assert result == []

    def test_negative_start_does_not_crash(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        result = ContextCompressor._find_context_summaries(messages, -10, 1)
        assert result == []

    def test_start_beyond_end_is_empty(self):
        messages = [{"role": "user", "content": "hello"}]
        result = ContextCompressor._find_context_summaries(messages, 50, 100)
        assert result == []

    def test_find_latest_context_summary_with_bad_bounds(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        idx, body = ContextCompressor._find_latest_context_summary(messages, 0, 999)
        assert idx is None
        assert body == ""


class TestCompressEndToEndShortToolSuffix:
    """compress()-level E2E regression for #75588.

    The unit tests above poke ``_find_tail_cut_by_tokens`` and
    ``_find_context_summaries`` directly; this class drives the REAL
    ``compress()`` pipeline over the exact live-shaped transcript from the
    issue: an 8-message session (system + tool-only suffix) where
    ``_align_boundary_forward`` slides the protected start to
    ``len(messages)`` and, pre-fix, ``_find_tail_cut_by_tokens`` returned
    ``len(messages) + 1`` via the ``head_end + 1`` forward-progress floor.

    With ``compress_start = n`` and ``compress_end = n + 1`` the
    ``compress_start >= compress_end`` no-op guard does NOT fire, so the
    out-of-range ``compress_end`` flows into the downstream summary scans —
    the IndexError observed live.  Post-fix the tail cut is clamped to ``n``,
    the guard fires, and compress() must: raise nothing, never call the
    summary LLM, and hand the transcript back unchanged.
    """

    def _live_shaped_transcript(self):
        """The reproduced internal bounds from #75588: len(messages)=8,
        system prompt + a tool-only suffix (mid-run gateway hygiene shape —
        the parent assistant tool_calls turn was already summarized away)."""
        return [{"role": "system", "content": "sys"}] + [
            {"role": "tool", "content": f"result {i}", "tool_call_id": "tc1"}
            for i in range(7)
        ]

    def test_compress_short_tool_suffix_no_crash_no_llm_unchanged(self, compressor):
        import copy

        c = compressor
        # A previously-compacted session: protect_first_n decays to 0, so the
        # protected head is just the system prompt; the forward alignment then
        # walks the tool-only suffix to head_end == len(messages) == 8, and
        # pre-fix the tail cut returned 9.
        c.compression_count = 1
        messages = self._live_shaped_transcript()
        snapshot = copy.deepcopy(messages)

        # Sanity: this transcript really produces the boundary shape from the
        # issue (start == n; end must be clamped to n, pre-fix it was n + 1).
        n = len(messages)
        start = c._align_boundary_forward(messages, c._protect_head_size(messages))
        assert start == n, f"fixture drifted: aligned start {start} != n {n}"
        assert c._find_tail_cut_by_tokens(messages, start) == n

        with patch.object(
            c, "_generate_summary", return_value="SHOULD NOT BE CALLED"
        ) as gen:
            # Must not raise (pre-fix: IndexError scanning past the end of
            # the list with the unclamped compress_end).
            out = c.compress(messages, current_tokens=90_000)

        assert gen.call_count == 0, (
            "compress() invoked the summary LLM even though there is no "
            "compressible window (compress_start == len(messages)); the "
            "clamped tail cut should make the no-op guard fire instead."
        )
        assert out == snapshot, (
            "compress() must return the transcript unchanged when the "
            f"protected head covers the whole list. Got: {out}"
        )
