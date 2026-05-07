"""Boundary atomicity for compaction: parallel tool calls + first-assistant anchor."""

from agent.context_compressor import ContextCompressor


def _bare_compressor(**overrides):
    c = ContextCompressor.__new__(ContextCompressor)
    c.protect_first_n = 1
    c.protect_last_n = 2
    c.tail_token_budget = 5000
    c.context_length = 200_000
    c.threshold_percent = 0.50
    c.threshold_tokens = 100_000
    c.anchor_first_assistant = True
    c.quiet_mode = True
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


class TestFirstAssistantAnchor:
    def test_anchor_skips_leading_user_block(self):
        """If protect_first_n=1 lands on a user msg, anchor slides to first assistant."""
        c = _bare_compressor(protect_first_n=1)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},      # idx 1, would be start
            {"role": "user", "content": "u2"},      # idx 2
            {"role": "assistant", "content": "a1"}, # idx 3 — should anchor here
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a2"},
        ]
        anchored = c._anchor_to_first_assistant(msgs, start_idx=1)
        assert anchored == 3

    def test_anchor_no_op_when_already_at_assistant(self):
        c = _bare_compressor(protect_first_n=2)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"}, # idx 2, start_idx=2
            {"role": "user", "content": "u2"},
        ]
        assert c._anchor_to_first_assistant(msgs, start_idx=2) == 2

    def test_anchor_does_not_cross_tail(self):
        """If no assistant exists between start and tail, return unchanged."""
        c = _bare_compressor()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "user", "content": "u3"},  # tail starts here
        ]
        # Caller's logic still handles "no compress region" — anchor just doesn't lie.
        result = c._anchor_to_first_assistant(msgs, start_idx=1, tail_start=3)
        assert result >= 3  # signals "nothing to anchor"; caller must check

    def test_anchor_disabled_when_flag_off(self):
        c = _bare_compressor(anchor_first_assistant=False)
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        # With flag off, returns start_idx unchanged
        assert c._anchor_to_first_assistant(msgs, start_idx=0) == 0


class TestParallelToolCallsAtomicity:
    """Existing helpers must keep parallel tool_call/result groups together."""

    def test_aligned_backward_pulls_assistant_with_parallel_results(self):
        """Tail boundary inside a 3-tool-result block must walk back to the assistant."""
        c = _bare_compressor()
        msgs = [
            {"role": "user", "content": "do three things"},
            {"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "2", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "3", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "r1"},
            {"role": "tool", "tool_call_id": "2", "content": "r2"},  # boundary lands here
            {"role": "tool", "tool_call_id": "3", "content": "r3"},
            {"role": "user", "content": "thanks"},
        ]
        # Call boundary at idx=4 (between r2 and r3). Must walk back to idx=1 (assistant).
        aligned = c._align_boundary_backward(msgs, idx=4)
        assert aligned == 1, (
            f"Expected boundary to pull back to assistant at idx=1, got {aligned}. "
            f"Splitting parallel tool_results would orphan tool_call_ids."
        )

    def test_sanitize_removes_orphan_results_from_split_parallel_group(self):
        """If sanitization sees a tool result whose call_id was summarized away, drop it."""
        c = _bare_compressor()
        # Compressed list missing the assistant with tool_calls
        compressed = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "summary placeholder"},
            {"role": "tool", "tool_call_id": "orphan_1", "content": "stale result"},
            {"role": "user", "content": "next question"},
        ]
        out = c._sanitize_tool_pairs(compressed)
        roles = [m.get("role") for m in out]
        assert "tool" not in roles, "Orphaned tool result must be dropped"
