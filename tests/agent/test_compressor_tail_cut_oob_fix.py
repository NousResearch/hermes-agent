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
