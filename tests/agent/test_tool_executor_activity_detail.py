"""Invariant: the running tool's preview and start time are recorded for the heartbeat.

``_begin_tool_execution`` already knows the tool's redacted display args (it builds the progress
preview from them). The heartbeat reads ``_current_tool_detail`` / ``_current_tool_started`` off the
agent, so the recording has to happen there — a heartbeat that silently loses its detail is the
regression this covers.
"""

import time
from unittest.mock import MagicMock

import pytest


def _agent():
    agent = MagicMock()
    agent.quiet_mode = True          # skip the printed progress lines
    agent.verbose_logging = False
    agent.tool_progress_callback = None
    agent.log_prefix_chars = 80
    agent._checkpoint_mgr.enabled = False
    return agent


def test_begin_tool_execution_records_preview_and_start_time():
    from agent.tool_executor import _ToolCallRef, _begin_tool_execution

    agent = _agent()
    ref = _ToolCallRef("terminal", {"command": "git status --short"}, "task-1", "call-1", None)
    before = time.time()
    _begin_tool_execution(agent, ref, None)

    assert agent._current_tool == "terminal"
    assert "git status" in (agent._current_tool_detail or "")
    assert before <= agent._current_tool_started <= time.time()


def test_begin_tool_execution_records_no_detail_for_unpreviewable_args():
    """A custom tool with no previewable argument must record ``None``, never ``"None"`` text."""
    from agent.tool_executor import _ToolCallRef, _begin_tool_execution

    agent = _agent()
    ref = _ToolCallRef("some_custom_tool", {}, "task-1", "call-2", None)
    _begin_tool_execution(agent, ref, None)

    assert agent._current_tool == "some_custom_tool"
    assert agent._current_tool_detail is None
