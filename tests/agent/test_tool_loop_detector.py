"""Tests for tool call loop detection.

Verifies:
1. Identical (tool, args) repeated N times triggers generic_repeat
2. Same tool with different args but identical results triggers poll_no_progress
3. Alternating A-B-A-B pattern triggers ping_pong
4. Different tools or different args do NOT trigger false positives
5. Window slides correctly — old entries expire
6. Severity escalates: no_detection → warning → critical
7. reset() clears state
"""
import pytest

from agent.tool_loop_detector import ToolLoopDetector, LoopVerdict


class TestGenericRepeat:
    """Same (tool_name, canonical_args) called N times."""

    def test_no_detection_below_threshold(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        for _ in range(2):
            v = d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3 matches")
        assert v.severity == "none"

    def test_warning_at_threshold(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        for _ in range(2):
            d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3 matches")
        v = d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3 matches")
        assert v.severity == "warning"
        assert v.detector == "generic_repeat"
        assert v.streak == 3

    def test_critical_at_threshold(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        for _ in range(4):
            d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3 matches")
        v = d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3 matches")
        assert v.severity == "critical"
        assert v.streak == 5

    def test_different_args_resets_streak(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3")
        d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3")
        d.record("grep", {"pattern": "bar", "path": "/src"}, result="found 1")
        v = d.record("grep", {"pattern": "foo", "path": "/src"}, result="found 3")
        assert v.severity == "none"

    def test_different_tool_resets_streak(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        d.record("grep", {"pattern": "foo"}, result="x")
        d.record("grep", {"pattern": "foo"}, result="x")
        d.record("read_file", {"path": "/tmp/a.py"}, result="content")
        v = d.record("grep", {"pattern": "foo"}, result="x")
        assert v.severity == "none"

    def test_arg_order_irrelevant(self):
        """Args {a:1, b:2} and {b:2, a:1} are the same call."""
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        d.record("t", {"a": 1, "b": 2}, result="r")
        d.record("t", {"b": 2, "a": 1}, result="r")
        v = d.record("t", {"a": 1, "b": 2}, result="r")
        assert v.severity == "warning"
        assert v.streak == 3


class TestPollNoProgress:
    """Same tool, possibly different args, but identical results N times."""

    def test_same_tool_identical_results_different_args(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        d.record("process_poll", {"pid": "123"}, result='{"status":"running"}')
        d.record("process_poll", {"pid": "123", "verbose": True}, result='{"status":"running"}')
        v = d.record("process_poll", {"pid": "123"}, result='{"status":"running"}')
        assert v.severity == "warning"
        assert v.detector == "poll_no_progress"

    def test_different_results_no_detection(self):
        """Same tool, varying args, different results — no loop."""
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        d.record("process_poll", {"pid": "123", "seq": 1}, result='{"status":"running","progress":10}')
        d.record("process_poll", {"pid": "123", "seq": 2}, result='{"status":"running","progress":20}')
        v = d.record("process_poll", {"pid": "123", "seq": 3}, result='{"status":"running","progress":30}')
        assert v.severity == "none"


class TestPingPong:
    """Alternating between exactly 2 tool states: A-B-A-B."""

    def test_abab_detected(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        d.record("read_file", {"path": "/a.py"}, result="content_a")
        d.record("write_file", {"path": "/a.py", "content": "fix"}, result="ok")
        d.record("read_file", {"path": "/a.py"}, result="content_a")
        d.record("write_file", {"path": "/a.py", "content": "fix"}, result="ok")
        d.record("read_file", {"path": "/a.py"}, result="content_a")
        v = d.record("write_file", {"path": "/a.py", "content": "fix"}, result="ok")
        assert v.severity == "warning"
        assert v.detector == "ping_pong"

    def test_abc_not_detected(self):
        """Three different tools cycling is not a ping-pong."""
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        for _ in range(3):
            d.record("a", {}, result="1")
            d.record("b", {}, result="2")
            d.record("c", {}, result="3")
        v = d.record("a", {}, result="1")
        assert v.detector != "ping_pong" or v.severity == "none"


class TestWindowSliding:
    """Old entries expire beyond window_size."""

    def test_entries_beyond_window_forgotten(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5, window_size=5)
        d.record("grep", {"pattern": "foo"}, result="r")
        d.record("grep", {"pattern": "foo"}, result="r")
        for i in range(5):
            d.record(f"tool_{i}", {}, result="x")
        v = d.record("grep", {"pattern": "foo"}, result="r")
        assert v.severity == "none"


class TestReasoningIntentExtraction:
    """Extract intended tool name from reasoning text."""

    def test_extracts_intended_tool(self):
        d = ToolLoopDetector(
            warning_threshold=3, critical_threshold=5,
            valid_tool_names={"mcp_n8n_n8n_generate_workflow", "mcp_n8n_n8n_test_workflow"},
        )
        for _ in range(2):
            d.record("mcp_n8n_n8n_generate_workflow", {"description": "test"}, result="error")
        v = d.record(
            "mcp_n8n_n8n_generate_workflow",
            {"description": "test"},
            result="error",
            reasoning="I want to call mcp_n8n_n8n_test_workflow but keep hitting generate."
        )
        assert v.severity == "warning"
        assert v.intended_tool == "mcp_n8n_n8n_test_workflow"

    def test_no_reasoning_no_intended_tool(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        for _ in range(2):
            d.record("grep", {"pattern": "foo"}, result="r")
        v = d.record("grep", {"pattern": "foo"}, result="r")
        assert v.intended_tool is None


class TestReset:
    def test_reset_clears_state(self):
        d = ToolLoopDetector(warning_threshold=3, critical_threshold=5)
        for _ in range(4):
            d.record("grep", {"pattern": "foo"}, result="r")
        d.reset()
        v = d.record("grep", {"pattern": "foo"}, result="r")
        assert v.severity == "none"
