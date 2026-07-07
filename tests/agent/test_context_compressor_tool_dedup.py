"""Tests for duplicate tool_call_id sanitization in compression (#58327)."""

import pytest
from unittest.mock import MagicMock

from agent.context_compressor import CompactionContext


class TestSanitizeDuplicateToolCallIds:
    """Compression sanitizer must dedup duplicate tool_call_ids.

    After compression/SMR, a single tool_call_id can appear in multiple
    tool messages (original result survives in the tail while the summary
    re-inserts a back-reference).  Strict providers reject HTTP 400
    "Duplicate value for 'tool_call_id'" (#58327).
    """

    @pytest.fixture
    def ctx(self):
        cc = CompactionContext.__new__(CompactionContext)
        cc.quiet_mode = False
        return cc

    def test_removes_duplicate_tool_call_ids_keeps_last(self, ctx):
        messages = [
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": "ls"}},
                {"id": "call_2", "function": {"name": "terminal", "arguments": "pwd"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file1 file2"},
            {"role": "tool", "tool_call_id": "call_2", "content": "/home"},
            {"role": "tool", "tool_call_id": "call_1", "content": "file1 file2"},  # duplicate
        ]
        result = ctx._sanitize_tool_pairs(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        call_ids = [m.get("tool_call_id") for m in tool_msgs]
        # call_2 should be preserved, call_1 should appear exactly once
        assert call_ids.count("call_1") == 1
        assert call_ids.count("call_2") == 1
        assert len(tool_msgs) == 2

    def test_removes_multiple_duplicates(self, ctx):
        messages = [
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file", "arguments": "x.py"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "content v1"},
            {"role": "tool", "tool_call_id": "call_1", "content": "content v2"},
            {"role": "tool", "tool_call_id": "call_1", "content": "content v3"},
        ]
        result = ctx._sanitize_tool_pairs(messages)
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        # Only the last occurrence survives
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "content v3"
        assert tool_msgs[0]["tool_call_id"] == "call_1"

    def test_no_duplicates_no_change(self, ctx):
        messages = [
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "output"},
        ]
        result = ctx._sanitize_tool_pairs(messages)
        assert len(result) == 3

    def test_duplicates_with_orphans_both_handled(self, ctx):
        """Dedup AND orphan removal should work together."""
        messages = [
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": "ls"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "output"},
            {"role": "tool", "tool_call_id": "call_1", "content": "output"},    # duplicate
            {"role": "tool", "tool_call_id": "call_orphan", "content": "orphan"}, # orphan
        ]
        result = ctx._sanitize_tool_pairs(messages)
        tool_ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
        assert tool_ids == ["call_1"]

    def test_non_tool_messages_preserved(self, ctx):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = ctx._sanitize_tool_pairs(messages)
        assert result == messages
