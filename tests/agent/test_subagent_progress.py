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
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

from agent.display import KawaiiSpinner
from tools.delegate_tool import _build_child_progress_callback


# =========================================================================
# KawaiiSpinner.print_above tests
# =========================================================================

class TestPrintAbove:
    """Tests for KawaiiSpinner.print_above method."""

    def test_print_above_without_spinner_running(self):
        """print_above should write to stdout even when spinner is not running."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf  # Redirect to buffer
        
        spinner.print_above("hello world")
        output = buf.getvalue()
        assert "hello world" in output

    def test_print_above_with_spinner_running(self):
        """print_above should clear spinner line and print text."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("test")
        spinner._out = buf
        spinner.running = True  # Pretend spinner is running (don't start thread)
        
        spinner.print_above("tool line")
        output = buf.getvalue()
        assert "tool line" in output
        assert "\r\033[K" in output  # Should start with line clear

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

    def test_returns_none_when_no_display(self):
        """Should return None when parent has no spinner or callback."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent.tool_progress_callback = None
        
        cb = _build_child_progress_callback(0, parent)
        assert cb is None

    def test_cli_spinner_tool_event(self):
        """Should print tool line above spinner for CLI path."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("delegating")
        spinner._out = buf
        spinner.running = True
        
        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None
        
        cb = _build_child_progress_callback(0, parent)
        assert cb is not None
        
        cb("web_search", "quantum computing")
        output = buf.getvalue()
        assert "web_search" in output
        assert "quantum computing" in output
        assert "├─" in output

    def test_cli_spinner_thinking_event(self):
        """Should print thinking line above spinner for CLI path."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("delegating")
        spinner._out = buf
        spinner.running = True
        
        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None
        
        cb = _build_child_progress_callback(0, parent)
        cb("_thinking", "I'll search for papers first")
        
        output = buf.getvalue()
        assert "💭" in output
        assert "search for papers" in output

    def test_gateway_batched_progress(self):
        """Gateway path should batch tool calls and flush at BATCH_SIZE."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        
        cb = _build_child_progress_callback(0, parent)
        
        # Send 4 tool calls — shouldn't flush yet (BATCH_SIZE = 5)
        for i in range(4):
            cb(f"tool_{i}", f"arg_{i}")
        parent_cb.assert_not_called()
        
        # 5th call should trigger flush
        cb("tool_4", "arg_4")
        parent_cb.assert_called_once()
        call_args = parent_cb.call_args
        assert "tool_0" in call_args[0][1]
        assert "tool_4" in call_args[0][1]

    def test_thinking_not_relayed_to_gateway(self):
        """Thinking events should NOT be sent to gateway (too noisy)."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        
        cb = _build_child_progress_callback(0, parent)
        cb("_thinking", "some reasoning text")
        
        parent_cb.assert_not_called()

    def test_parallel_callbacks_independent(self):
        """Each child's callback should have independent batch state."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb
        
        cb0 = _build_child_progress_callback(0, parent)
        cb1 = _build_child_progress_callback(1, parent)
        
        # Send 3 calls to each — neither should flush (batch size = 5)
        for i in range(3):
            cb0(f"tool_{i}")
            cb1(f"other_{i}")
        
        parent_cb.assert_not_called()

    def test_task_index_prefix_in_output(self):
        """Multi-task mode should show task index prefix."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("delegating")
        spinner._out = buf
        spinner.running = True
        
        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None
        
        # task_index > 0 should add prefix
        cb = _build_child_progress_callback(2, parent)
        cb("web_search", "test")
        
        output = buf.getvalue()
        assert "[2]" in output

    def test_task_index_zero_no_prefix(self):
        """Single task (index 0) should not show index prefix."""
        buf = io.StringIO()
        spinner = KawaiiSpinner("delegating")
        spinner._out = buf
        spinner.running = True
        
        parent = MagicMock()
        parent._delegate_spinner = spinner
        parent.tool_progress_callback = None
        
        cb = _build_child_progress_callback(0, parent)
        cb("web_search", "test")
        
        output = buf.getvalue()
        assert "[0]" not in output


# =========================================================================
# Integration: thinking callback in run_agent.py
# =========================================================================

class TestThinkingCallback:
    """Tests for the _thinking callback in AIAgent conversation loop."""

    def _simulate_thinking_callback(self, content, callback, delegate_depth=1):
        """Simulate the exact code path from run_agent.py for the thinking callback.
        
        delegate_depth: simulates self._delegate_depth.
            0 = main agent (should NOT fire), >=1 = subagent (should fire).
        """
        import re
        if (content and callback and delegate_depth > 0):
            _think_text = content.strip()
            _think_text = re.sub(
                r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
            ).strip()
            first_line = _think_text.split('\n')[0][:80] if _think_text else ""
            if first_line:
                try:
                    callback("_thinking", first_line)
                except Exception:
                    pass

    def test_thinking_callback_fires_on_content(self):
        """tool_progress_callback should receive _thinking event
        when assistant message has content."""
        calls = []
        self._simulate_thinking_callback(
            "I'll research quantum computing first, then summarize.",
            lambda name, preview=None: calls.append((name, preview))
        )
        assert len(calls) == 1
        assert calls[0][0] == "_thinking"
        assert "quantum computing" in calls[0][1]

    def test_thinking_callback_skipped_when_no_content(self):
        """Should not fire when assistant has no content."""
        calls = []
        self._simulate_thinking_callback(
            None,
            lambda name, preview=None: calls.append((name, preview))
        )
        assert len(calls) == 0

    def test_thinking_callback_truncates_long_content(self):
        """Should truncate long content to 80 chars."""
        calls = []
        self._simulate_thinking_callback(
            "A" * 200 + "\nSecond line should be ignored",
            lambda name, preview=None: calls.append((name, preview))
        )
        assert len(calls) == 1
        assert len(calls[0][1]) == 80

    def test_thinking_callback_skipped_for_main_agent(self):
        """Main agent (delegate_depth=0) should NOT fire thinking events.
        This prevents gateway spam on Telegram/Discord."""
        calls = []
        self._simulate_thinking_callback(
            "I'll help you with that request.",
            lambda name, preview=None: calls.append((name, preview)),
            delegate_depth=0,
        )
        assert len(calls) == 0

    def test_thinking_callback_strips_reasoning_scratchpad(self):
        """REASONING_SCRATCHPAD tags should be stripped before display."""
        calls = []
        self._simulate_thinking_callback(
            "<REASONING_SCRATCHPAD>I need to analyze this carefully</REASONING_SCRATCHPAD>",
            lambda name, preview=None: calls.append((name, preview))
        )
        assert len(calls) == 1
        assert "<REASONING_SCRATCHPAD>" not in calls[0][1]
        assert "analyze this carefully" in calls[0][1]

    def test_thinking_callback_strips_think_tags(self):
        """<think> tags should be stripped before display."""
        calls = []
        self._simulate_thinking_callback(
            "<think>Let me think about this problem</think>",
            lambda name, preview=None: calls.append((name, preview))
        )
        assert len(calls) == 1
        assert "<think>" not in calls[0][1]
        assert "think about this problem" in calls[0][1]

    def test_thinking_callback_empty_after_strip(self):
        """Should not fire when content is only XML tags."""
        calls = []
        self._simulate_thinking_callback(
            "<REASONING_SCRATCHPAD></REASONING_SCRATCHPAD>",
            lambda name, preview=None: calls.append((name, preview))
        )
        assert len(calls) == 0


# =========================================================================
# Gateway batch flush tests
# =========================================================================

class TestBatchFlush:
    """Tests for gateway batch flush on subagent completion."""

    def test_flush_sends_remaining_batch(self):
        """_flush should send remaining tool names to gateway."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, parent)

        # Send 3 tools (below batch size of 5)
        cb("web_search", "query1")
        cb("read_file", "file.txt")
        cb("write_file", "out.txt")
        parent_cb.assert_not_called()

        # Flush should send the remaining 3
        cb._flush()
        parent_cb.assert_called_once()
        summary = parent_cb.call_args[0][1]
        assert "web_search" in summary
        assert "write_file" in summary

    def test_flush_noop_when_batch_empty(self):
        """_flush should not send anything when batch is empty."""
        parent = MagicMock()
        parent._delegate_spinner = None
        parent_cb = MagicMock()
        parent.tool_progress_callback = parent_cb

        cb = _build_child_progress_callback(0, parent)
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

        cb = _build_child_progress_callback(0, parent)
        cb("web_search", "test")
        cb._flush()  # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

