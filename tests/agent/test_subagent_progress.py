"""
Tests for subagent progress relay (issue #169).

Verifies that:
- KawaiiSpinner.print_above() works with and without active spinner
- _build_child_progress_callback handles CLI/gateway/no-display paths
- Thinking events are relayed correctly
- Parallel callbacks don't share state
"""

import io
import sys
import pytest
from unittest.mock import MagicMock

from agent.display import KawaiiSpinner
from tools.delegate_tool import _build_child_progress_callback
from tools.delegate_tool_progress import CHILD_TOOL_COMPLETED


# =========================================================================
# KawaiiSpinner.print_above tests
# =========================================================================

class TestPrintAbove:
    """Tests for KawaiiSpinner.print_above method."""


    def test_print_above_with_spinner_running(self):
        """print_above should clear spinner line and print text."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        spinner.running = True  # Pretend spinner is running (don't start thread)
        
        spinner.print_above("tool line")
        output = buf.getvalue()
        assert "tool line" in output
        assert "\r" in output  # Should start with carriage return to clear spinner line

    def test_print_above_uses_captured_stdout(self):
        """print_above should use self._out, not sys.stdout.
        This ensures it works inside redirect_stdout(devnull)."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        
        # Simulate redirect_stdout(devnull)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            spinner.print_above("should go to buf")
        finally:
            sys.stdout = old_stdout
        
        assert "should go to buf" in buf.getvalue()


# =========================================================================
# _build_child_progress_callback tests
# =========================================================================

class TestBuildChildProgressCallback:
    """Tests for child progress callback builder."""




    def test_gateway_batched_progress(self):
        """Gateway path: each tool.started relays a subagent.tool event, and a
        subagent.progress summary fires once BATCH_SIZE tools accumulate."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, "test goal", parent)

        # Each tool.started relays a subagent.tool event immediately (per-tool relay).
        for i in range(4):
            cb("tool.started", f"tool_{i}", f"arg_{i}", {})
        # 4 per-tool relays so far, no batch summary yet (BATCH_SIZE=5)
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events == ["subagent.tool"] * 4

        # 5th call triggers another per-tool relay PLUS the batch-size summary
        cb("tool.started", "tool_4", "arg_4", {})
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events == ["subagent.tool"] * 5 + ["subagent.progress"]
        summary_call = parent_cb.call_args_list[-1]
        summary_text = summary_call.kwargs.get("preview") or summary_call.args[2]
        assert "tool_0" in summary_text
        assert "tool_4" in summary_text


    def test_parallel_callbacks_independent(self):
        """Each child's callback batches tool names independently."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb0 = _build_child_progress_callback(0, "goal a", parent)
        cb1 = _build_child_progress_callback(1, "goal b", parent)

        # 3 tool.started per child = 6 per-tool relays; neither should hit
        # the batch-size summary (batch size = 5, counted per-child).
        for i in range(3):
            cb0("tool.started", f"tool_{i}", f"a_{i}", {})
            cb1("tool.started", f"other_{i}", f"b_{i}", {})

        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events.count("subagent.tool") == 6
        assert "subagent.progress" not in events

    def test_task_index_prefix_in_batch_mode(self):
        """Batch mode (task_count > 1) should show 1-indexed prefix for all tasks."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("delegating")
        spinner._out = buf
        spinner.running = True
        
        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None
        
        # task_index=0 in a batch of 3 → prefix "[1/3]" (batch slot; a
        # delegation batch tag is prepended when the id is known)
        cb0 = _build_child_progress_callback(0, "test goal", parent, task_count=3)
        cb0("tool.started", "web_search", "test", {})
        output = buf.getvalue()
        assert "[1/3]" in output

        # task_index=2 in a batch of 3 → prefix "[3]"
        buf.truncate(0)
        buf.seek(0)
        cb2 = _build_child_progress_callback(2, "test goal", parent, task_count=3)
        cb2("tool.started", "web_search", "test", {})
        output = buf.getvalue()
        assert "[3/3]" in output



# =========================================================================
# Integration: thinking callback in run_agent.py
# =========================================================================







# =========================================================================
# Gateway batch flush tests
# =========================================================================

class TestBatchFlush:
    """Tests for gateway batch flush on subagent completion."""

    def test_flush_sends_remaining_batch(self):
        """_flush should send a final subagent.progress summary of any unsent
        tool names in the batch (less than BATCH_SIZE)."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, "test goal", parent)

        # Send 3 tools (below batch size of 5) — each relays subagent.tool
        cb("tool.started", "web_search", "query1", {})
        cb("tool.started", "read_file", "file.txt", {})
        cb("tool.started", "write_file", "out.txt", {})
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events == ["subagent.tool"] * 3  # per-tool relays so far
        assert "subagent.progress" not in events  # no batch-size summary yet

        # Flush should send the remaining 3 as a summary
        cb._flush()
        events = [c.args[0] for c in parent_cb.call_args_list]
        assert events[-1] == "subagent.progress"
        summary_call = parent_cb.call_args_list[-1]
        summary_text = summary_call.kwargs.get("preview") or summary_call.args[2]
        assert "web_search" in summary_text
        assert "write_file" in summary_text

    def test_flush_noop_when_batch_empty(self):
        """_flush should not send anything when batch is empty."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, "test goal", parent)
        cb._flush()
        parent_cb.assert_not_called()

    def test_flush_noop_when_no_parent_callback(self):
        """_flush should not crash when there's no parent callback."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        spinner.running = True

        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None

        cb = _build_child_progress_callback(0, "test goal", parent)
        cb("tool.started", "web_search", "test", {})
        cb._flush()  # Should not crash


# =========================================================================
# Child tool completion relay
# =========================================================================

class TestChildToolCompletion:
    """A child's tool.completed reaches the parent as its outcome only."""

    @staticmethod
    def _relay():
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        return _build_child_progress_callback(0, "test goal", parent, subagent_id="sa-1", depth=1), parent_cb

    def test_completion_relays_outcome_without_the_result(self):
        cb, parent_cb = self._relay()
        cb("tool.started", "terminal", "ls", {"command": "ls"})
        cb("tool.completed", "terminal", None, None, duration=1.5, is_error=True, result="result-marker")

        started, completed = parent_cb.call_args_list
        assert completed.args[0] == CHILD_TOOL_COMPLETED
        assert completed.args[1] == "terminal"
        assert completed.kwargs["duration"] == 1.5 and completed.kwargs["is_error"] is True
        assert completed.kwargs["subagent_id"] == "sa-1" and completed.kwargs["depth"] == 1
        assert completed.kwargs["tool_index"] == started.kwargs["tool_index"] == 1
        assert "result-marker" not in repr(completed)

    def test_completions_pair_with_starts_of_the_same_tool_in_order(self):
        cb, parent_cb = self._relay()
        cb("tool.started", "terminal", "a", {})
        cb("tool.started", "read_file", "b", {})
        cb("tool.started", "terminal", "c", {})
        cb("tool.completed", "terminal", None, None, duration=0.1, is_error=False)
        cb("tool.completed", "terminal", None, None, duration=0.1, is_error=False)

        done = [c.kwargs["tool_index"] for c in parent_cb.call_args_list if c.args[0] == CHILD_TOOL_COMPLETED]
        assert done == [1, 3]

    def test_completions_pair_by_call_id_when_they_finish_out_of_order(self):
        cb, parent_cb = self._relay()
        cb("tool.started", "terminal", "a", {}, tool_call_id="call-a")
        cb("tool.started", "terminal", "b", {}, tool_call_id="call-b")
        cb("tool.completed", "terminal", None, None, duration=0.1, is_error=False, tool_call_id="call-b")
        cb("tool.completed", "terminal", None, None, duration=0.1, is_error=True, tool_call_id="call-a")

        done = [(c.kwargs["tool_call_id"], c.kwargs["tool_index"])
                for c in parent_cb.call_args_list if c.args[0] == CHILD_TOOL_COMPLETED]
        assert done == [("call-b", 2), ("call-a", 1)]

    def test_tool_call_id_is_relayed_when_the_producer_supplies_one(self):
        cb, parent_cb = self._relay()
        cb("tool.started", "terminal", "ls", {}, tool_call_id="call-1")
        cb("tool.started", "terminal", "ls")
        cb("tool.completed", "terminal", None, None, duration=0.1, is_error=False, tool_call_id="call-1")

        started_a, started_b, completed = parent_cb.call_args_list
        assert started_a.kwargs["tool_call_id"] == completed.kwargs["tool_call_id"] == "call-1"
        assert "tool_call_id" not in started_b.kwargs

    def test_completion_without_parent_callback_does_not_crash(self):
        parent = MagicMock()
        parent._delegate_spinner = KawaiiSpinner("test")
        parent.tool_progress_callback = None
        cb = _build_child_progress_callback(0, "test goal", parent)
        cb("tool.started", "terminal", "ls", {})
        cb("tool.completed", "terminal", None, None, duration=0.1, is_error=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

