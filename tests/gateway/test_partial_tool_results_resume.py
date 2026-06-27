"""Regressions for partial tool-call results on session resume.

When a gateway process is killed after some but not all tool results are
persisted, the resumed session has an incomplete assistant→tool sequence
that causes HTTP 400 from the API.

Bug: _strip_dangling_tool_call_tail only handles the ZERO-results case.
Partial results fall through, causing permanent session brick on resume.
"""

import pytest
from gateway.run import (
    _repair_partial_tool_call_results,
    _strip_dangling_tool_call_tail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assistant_msg(tool_calls):
    """Build an assistant message with tool_calls."""
    return {"role": "assistant", "tool_calls": tool_calls}


def _tool_msg(call_id, content="ok"):
    """Build a tool result message."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _tc(call_id):
    """Build a minimal tool_call entry."""
    return {"id": call_id, "function": {"name": "test_tool", "arguments": "{}"}}


def _user_msg(content="hello"):
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Tests for _repair_partial_tool_call_results
# ---------------------------------------------------------------------------

class TestRepairPartialToolCallResults:
    """Partial tool-call results should be repaired with synthetic errors."""

    def test_no_tool_calls_noop(self):
        history = [_user_msg(), _assistant_msg([]), _tool_msg("c1")]
        assert _repair_partial_tool_call_results(history) == history

    def test_no_tail_assistant_noop(self):
        history = [_user_msg(), _assistant_msg([_tc("c1")]), _tool_msg("c1")]
        assert _repair_partial_tool_call_results(history) == history

    def test_zero_results_adds_synthetic(self):
        """Zero results — function adds synthetic results (strip handles it in pipeline)."""
        history = [_user_msg(), _assistant_msg([_tc("c1")])]
        result = _repair_partial_tool_call_results(history)
        assert len(result) == 3
        assert result[-1]["role"] == "tool"
        assert result[-1]["tool_call_id"] == "c1"

    def test_all_results_present_noop(self):
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
            _tool_msg("c2"),
        ]
        assert _repair_partial_tool_call_results(history) == history

    def test_one_missing_appends_synthetic(self):
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
        ]
        result = _repair_partial_tool_call_results(history)
        assert len(result) == 4
        assert result[-1]["role"] == "tool"
        assert result[-1]["tool_call_id"] == "c2"
        assert "lost" in result[-1]["content"].lower()

    def test_multiple_missing(self):
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2"), _tc("c3")]),
            _tool_msg("c2"),
        ]
        result = _repair_partial_tool_call_results(history)
        assert len(result) == 5  # 3 original + 2 synthetic (c1, c3)
        missing_ids = {m["tool_call_id"] for m in result[3:]}
        assert missing_ids == {"c1", "c3"}

    def test_no_list_copy(self):
        """Returned list should be a new list, not mutate the original."""
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
        ]
        original_len = len(history)
        result = _repair_partial_tool_call_results(history)
        assert len(history) == original_len
        assert result is not history

    def test_only_trailing_tool_results_counted(self):
        """Tool results before the assistant message should not fill gaps."""
        history = [
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
            _assistant_msg([_tc("c1"), _tc("c2")]),
        ]
        result = _repair_partial_tool_call_results(history)
        # The second assistant (index 2) has c1+c2, but no tool results follow it.
        # The tool result at index 1 belongs to the FIRST assistant.
        assert len(result) == 5  # 3 original + 2 synthetic (c1, c2)
        assert result[-1]["tool_call_id"] == "c2"

    def test_empty_history_noop(self):
        assert _repair_partial_tool_call_results([]) == []

    def test_none_tool_call_id_skipped(self):
        tc_no_id = {"function": {"name": "test_tool", "arguments": "{}"}}
        history = [
            _user_msg(),
            {"role": "assistant", "tool_calls": [_tc("c1"), tc_no_id]},
            _tool_msg("c1"),
        ]
        result = _repair_partial_tool_call_results(history)
        assert len(result) == 3  # only c1 expected, no missing


# ---------------------------------------------------------------------------
# Tests for _strip_dangling_tool_call_tail (existing function, verify no regressions)
# ---------------------------------------------------------------------------

class TestStripDanglingToolCallTail:
    """Existing zero-results stripping should not regress."""

    def test_zero_results_strips_tail(self):
        history = [_user_msg(), _assistant_msg([_tc("c1")])]
        result = _strip_dangling_tool_call_tail(history)
        assert len(result) == 1

    def test_partial_results_not_stripped(self):
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
        ]
        result = _strip_dangling_tool_call_tail(history)
        assert len(result) == 3

    def test_all_results_not_stripped(self):
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1")]),
            _tool_msg("c1"),
        ]
        result = _strip_dangling_tool_call_tail(history)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Integration: strip + repair pipeline
# ---------------------------------------------------------------------------

class TestStripThenRepairPipeline:
    """End-to-end: _strip_dangling_tool_call_tail → _repair_partial_tool_call_results."""

    def test_zero_results_stripped_before_repair(self):
        """Zero results → strip removes the tail, repair has nothing to do."""
        history = [_user_msg(), _assistant_msg([_tc("c1")])]
        stripped = _strip_dangling_tool_call_tail(history)
        repaired = _repair_partial_tool_call_results(stripped)
        assert len(repaired) == 1

    def test_partial_results_repaired_after_strip(self):
        """Partial results → strip doesn't fire, repair fills gaps."""
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
        ]
        stripped = _strip_dangling_tool_call_tail(history)
        assert len(stripped) == 3  # strip did not fire
        repaired = _repair_partial_tool_call_results(stripped)
        assert len(repaired) == 4
        assert repaired[-1]["tool_call_id"] == "c2"

    def test_complete_sequence_unmodified(self):
        history = [
            _user_msg(),
            _assistant_msg([_tc("c1"), _tc("c2")]),
            _tool_msg("c1"),
            _tool_msg("c2"),
        ]
        stripped = _strip_dangling_tool_call_tail(history)
        repaired = _repair_partial_tool_call_results(stripped)
        assert len(repaired) == 4

    def test_multiple_tool_call_groups(self):
        """Only the TRAILING assistant message is checked for gaps."""
        history = [
            _assistant_msg([_tc("a1")]),
            _tool_msg("a1"),
            _assistant_msg([_tc("b1"), _tc("b2")]),
            _tool_msg("b1"),
        ]
        stripped = _strip_dangling_tool_call_tail(history)
        repaired = _repair_partial_tool_call_results(stripped)
        assert len(repaired) == 5  # 4 original + 1 synthetic (b2)
        assert repaired[-1]["tool_call_id"] == "b2"
