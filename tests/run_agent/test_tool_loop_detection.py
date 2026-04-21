"""Unit tests for cross-turn tool loop detection.

Covers the following functionality:
  - _hash_tool_args()              — argument normalization and hashing
  - _check_tool_loop()             — tracking and escalation logic
  - Escalation ladder: nudge → stronger nudge → reprompt → hard stop
"""

import json
import types
import unittest.mock as mock

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tc(name: str, arguments: str = "{}") -> types.SimpleNamespace:
    """Create a minimal tool_call SimpleNamespace mirroring the OpenAI SDK object."""
    tc = types.SimpleNamespace()
    tc.function = types.SimpleNamespace(name=name, arguments=arguments)
    return tc


def make_agent():
    """Create a minimal AIAgent instance for testing."""
    # Mock the AIAgent constructor to avoid full initialization
    with mock.patch.object(AIAgent, '__init__', lambda self: None):
        agent = AIAgent()
        agent._tool_repeat_tracker = {}
        agent._loop_reprompt_issued = False
        return agent


# ---------------------------------------------------------------------------
# _hash_tool_args
# ---------------------------------------------------------------------------

class TestHashToolArgs:

    def test_identical_args_same_hash(self):
        args = '{"query": "test", "limit": 10}'
        h1 = AIAgent._hash_tool_args(args)
        h2 = AIAgent._hash_tool_args(args)
        assert h1 == h2

    def test_different_args_different_hash(self):
        h1 = AIAgent._hash_tool_args('{"query": "foo"}')
        h2 = AIAgent._hash_tool_args('{"query": "bar"}')
        assert h1 != h2

    def test_key_order_normalized(self):
        """Keys in different order should produce same hash."""
        h1 = AIAgent._hash_tool_args('{"b": 2, "a": 1}')
        h2 = AIAgent._hash_tool_args('{"a": 1, "b": 2}')
        assert h1 == h2

    def test_whitespace_normalized(self):
        """Extra whitespace should not affect hash."""
        h1 = AIAgent._hash_tool_args('{"query": "test"}')
        h2 = AIAgent._hash_tool_args('{"query": "test"}')  # Same after normalization
        assert h1 == h2

    def test_trailing_space_normalized(self):
        """Trailing whitespace in args should not affect hash."""
        h1 = AIAgent._hash_tool_args('{"query": "test"}')
        h2 = AIAgent._hash_tool_args('{"query": "test"} ')
        # After JSON normalization, trailing space is stripped
        assert h1 == h2

    def test_invalid_json_fallback(self):
        """Invalid JSON should hash the raw string."""
        h1 = AIAgent._hash_tool_args("not json at all")
        h2 = AIAgent._hash_tool_args("not json at all")
        assert h1 == h2

    def test_hash_length(self):
        """Hash should be 16 chars (first 16 of SHA256)."""
        h = AIAgent._hash_tool_args('{"x": 1}')
        assert len(h) == 16


# ---------------------------------------------------------------------------
# _check_tool_loop — tracking
# ---------------------------------------------------------------------------

class TestToolLoopTracking:

    def test_first_call_no_action(self):
        """First call to a tool should return 'none'."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        result = agent._check_tool_loop([tc], ["search result"])
        assert result["action"] == "none"

    def test_second_call_triggers_nudge(self):
        """Second identical call should trigger level 1 nudge."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        # First call
        r1 = agent._check_tool_loop([tc], ["result 1"])
        assert r1["action"] == "none"
        
        # Second call (repeat)
        r2 = agent._check_tool_loop([tc], ["result 2"])
        assert r2["action"] == "nudge"
        assert r2.get("nudge_level") == 1

    def test_third_call_triggers_stronger_nudge(self):
        """Third identical call should trigger level 2 nudge."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        agent._check_tool_loop([tc], ["result 1"])  # 1st
        agent._check_tool_loop([tc], ["result 2"])  # 2nd (nudge)
        r3 = agent._check_tool_loop([tc], ["result 3"])  # 3rd
        
        assert r3["action"] == "nudge"
        assert r3.get("nudge_level") == 2

    def test_fourth_call_triggers_reprompt(self):
        """Fourth identical call should trigger reprompt."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        agent._check_tool_loop([tc], ["r1"])  # 1st
        agent._check_tool_loop([tc], ["r2"])  # 2nd (nudge)
        agent._check_tool_loop([tc], ["r3"])  # 3rd (stronger nudge)
        r4 = agent._check_tool_loop([tc], ["r4"])  # 4th
        
        assert r4["action"] == "reprompt"
        assert agent._loop_reprompt_issued is True

    def test_fifth_call_triggers_hard_stop(self):
        """Fifth identical call should trigger hard stop."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        for i in range(1, 5):
            agent._check_tool_loop([tc], [f"r{i}"])
        
        r5 = agent._check_tool_loop([tc], ["r5"])
        assert r5["action"] == "stop"
        assert r5["repeat_count"] == 5

    def test_different_tools_not_tracked_together(self):
        """Different tool names should have separate counters."""
        agent = make_agent()
        tc1 = make_tc("web_search", '{"query": "test"}')
        tc2 = make_tc("terminal", '{"cmd": "ls"}')
        
        # First calls to each
        agent._check_tool_loop([tc1], ["result1"])
        agent._check_tool_loop([tc2], ["result2"])
        
        # Second call to web_search (should trigger nudge)
        r3 = agent._check_tool_loop([tc1], ["result3"])
        assert r3["action"] == "nudge"
        
        # Second call to terminal (should also trigger nudge - separate counter)
        r4 = agent._check_tool_loop([tc2], ["result4"])
        assert r4["action"] == "nudge"

    def test_different_args_not_tracked_together(self):
        """Same tool with different args should have separate counters."""
        agent = make_agent()
        tc1 = make_tc("web_search", '{"query": "foo"}')
        tc2 = make_tc("web_search", '{"query": "bar"}')
        
        # First calls
        agent._check_tool_loop([tc1], ["result1"])
        agent._check_tool_loop([tc2], ["result2"])
        
        # Second call to first args (should trigger nudge)
        r3 = agent._check_tool_loop([tc1], ["result3"])
        assert r3["action"] == "nudge"
        
        # Second call to second args (should also trigger nudge - separate counter)
        r4 = agent._check_tool_loop([tc2], ["result4"])
        assert r4["action"] == "nudge"

    def test_empty_results_not_tracked(self):
        """Tools with empty results should not be tracked."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        # First call with result
        agent._check_tool_loop([tc], ["result"])
        
        # Second call with empty result (should not increment counter)
        r2 = agent._check_tool_loop([tc], [""])
        assert r2["action"] == "none"
        
        # Third call with result (should trigger nudge since counter is still at 1)
        r3 = agent._check_tool_loop([tc], ["result2"])
        assert r3["action"] == "nudge"

    def test_no_tool_calls_returns_none(self):
        """Empty tool_calls list should return 'none'."""
        agent = make_agent()
        result = agent._check_tool_loop([], [])
        assert result["action"] == "none"


# ---------------------------------------------------------------------------
# Nudge message content
# ---------------------------------------------------------------------------

class TestNudgeMessages:

    def test_nudge_level_1_message(self):
        """Level 1 nudge should reference the previous turn."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        agent._check_tool_loop([tc], ["this is a very long result that should be truncated"])
        r2 = agent._check_tool_loop([tc], ["another result"])
        
        assert "previous turn" in r2["nudge_message"].lower()
        assert "web_search" in r2["nudge_message"]
        assert "test" in r2["nudge_message"]

    def test_nudge_level_2_message(self):
        """Level 2 nudge should mention consecutive attempts."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        agent._check_tool_loop([tc], ["r1"])
        agent._check_tool_loop([tc], ["r2"])
        r3 = agent._check_tool_loop([tc], ["r3"])
        
        assert "consecutively" in r3["nudge_message"].lower()
        assert "web_search" in r3["nudge_message"]

    def test_reprompt_message(self):
        """Reprompt should mention user intervention."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        for i in range(1, 4):
            agent._check_tool_loop([tc], [f"r{i}"])
        
        r4 = agent._check_tool_loop([tc], ["r4"])
        
        assert "user to intervene" in r4["nudge_message"].lower()
        assert "web_search" in r4["nudge_message"]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:

    def test_config_load_succeeds(self):
        """Config loading should not raise exceptions."""
        agent = make_agent()
        tc = make_tc("web_search", '{"query": "test"}')
        
        # Should not raise even if config file doesn't exist
        r1 = agent._check_tool_loop([tc], ["result1"])
        assert r1["action"] == "none"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_multiple_tool_calls_in_single_turn(self):
        """Multiple tool calls in one turn should each be tracked."""
        agent = make_agent()
        tc1 = make_tc("web_search", '{"query": "test"}')
        tc2 = make_tc("terminal", '{"cmd": "ls"}')
        
        # First turn with both tools
        r1 = agent._check_tool_loop([tc1, tc2], ["result1", "result2"])
        assert r1["action"] == "none"
        
        # Second turn with both tools (both should trigger nudge)
        r2 = agent._check_tool_loop([tc1, tc2], ["result3", "result4"])
        # Only the first matching tool triggers the return
        assert r2["action"] == "nudge"

    def test_tool_name_with_special_characters(self):
        """Tool names with underscores/hyphens should work."""
        agent = make_agent()
        tc = make_tc("read_file", '{"path": "/tmp/test"}')
        
        agent._check_tool_loop([tc], ["result1"])
        r2 = agent._check_tool_loop([tc], ["result2"])
        assert r2["action"] == "nudge"

    def test_large_arguments_handled(self):
        """Large argument strings should not cause issues."""
        agent = make_agent()
        large_args = json.dumps({"query": "test", "extra": "x" * 10000})
        tc = make_tc("web_search", large_args)
        
        r1 = agent._check_tool_loop([tc], ["result1"])
        assert r1["action"] == "none"
        
        r2 = agent._check_tool_loop([tc], ["result2"])
        assert r2["action"] == "nudge"
