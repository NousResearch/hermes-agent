"""Tests for agent.utils -- standalone utility functions extracted from run_agent.py.

These tests verify that functions extracted from run_agent.py into agent/utils.py
work correctly in their new home.
"""

import json
import pytest


class TestSanitizeSurrogates:
    def test_import(self):
        from agent.utils import _sanitize_surrogates
        assert callable(_sanitize_surrogates)

    def test_strips_lone_surrogates(self):
        from agent.utils import _sanitize_surrogates
        result = _sanitize_surrogates("hello\ud800world")
        assert "\ud800" not in result
        assert "hello" in result and "world" in result

    def test_preserves_normal_text(self):
        from agent.utils import _sanitize_surrogates
        assert _sanitize_surrogates("hello world") == "hello world"

    def test_handles_empty(self):
        from agent.utils import _sanitize_surrogates
        assert _sanitize_surrogates("") == ""


class TestRepairToolCallArguments:
    def test_import(self):
        from agent.utils import _repair_tool_call_arguments
        assert callable(_repair_tool_call_arguments)

    def test_fixes_truncated_json(self):
        from agent.utils import _repair_tool_call_arguments
        # Truncated JSON: missing closing brace
        result = _repair_tool_call_arguments('{"path": "/tmp/test"', "read_file")
        assert json.loads(result)  # Should be valid JSON
        assert json.loads(result)["path"] == "/tmp/test"

    def test_passes_valid_json_through(self):
        from agent.utils import _repair_tool_call_arguments
        valid = '{"path": "/tmp/test"}'
        result = _repair_tool_call_arguments(valid, "read_file")
        # Function normalizes to compact JSON
        assert json.loads(result) == json.loads(valid)

    def test_handles_empty_string(self):
        from agent.utils import _repair_tool_call_arguments
        result = _repair_tool_call_arguments("", "test")
        parsed = json.loads(result)
        assert parsed is not None  # Returns valid JSON (empty object)


class TestStripNonAscii:
    def test_import(self):
        from agent.utils import _strip_non_ascii
        assert callable(_strip_non_ascii)

    def test_strips_non_ascii(self):
        from agent.utils import _strip_non_ascii
        result = _strip_non_ascii("hello\x80world")
        assert "\x80" not in result

    def test_preserves_normal_text(self):
        from agent.utils import _strip_non_ascii
        assert _strip_non_ascii("hello world") == "hello world"


class TestSanitizeMessagesSurrogates:
    def test_import(self):
        from agent.utils import _sanitize_messages_surrogates
        assert callable(_sanitize_messages_surrogates)

    def test_cleans_message_content(self):
        from agent.utils import _sanitize_messages_surrogates
        msgs = [{"role": "user", "content": "hello\ud800world"}]
        _sanitize_messages_surrogates(msgs)
        assert "\ud800" not in msgs[0]["content"]

    def test_cleans_tool_call_arguments(self):
        from agent.utils import _sanitize_messages_surrogates
        msgs = [{"role": "assistant", "tool_calls": [{
            "id": "1", "type": "function",
            "function": {"name": "test", "arguments": '{"a":"b"}'}
        }]}]
        _sanitize_messages_surrogates(msgs)
        # Should still be valid JSON
        assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])


class TestIsDestructiveCommand:
    def test_import(self):
        from agent.utils import _is_destructive_command
        assert callable(_is_destructive_command)

    def test_rm_is_destructive(self):
        from agent.utils import _is_destructive_command
        assert _is_destructive_command("rm -rf /")

    def test_ls_not_destructive(self):
        from agent.utils import _is_destructive_command
        assert not _is_destructive_command("ls -la")

    def test_drop_is_destructive(self):
        from agent.utils import _is_destructive_command
        assert _is_destructive_command("truncate table users")


class TestShouldParallelizeToolBatch:
    def test_import(self):
        from agent.utils import _should_parallelize_tool_batch
        assert callable(_should_parallelize_tool_batch)

    def test_single_call_no_parallelize(self):
        from agent.utils import _should_parallelize_tool_batch
        from unittest.mock import MagicMock
        tc = MagicMock()
        tc.function.name = "read_file"
        assert not _should_parallelize_tool_batch([tc])

    def test_same_safe_tool_parallelize(self):
        from agent.utils import _should_parallelize_tool_batch
        from unittest.mock import MagicMock
        tcs = []
        for _ in range(3):
            tc = MagicMock()
            tc.function.name = "search_files"
            tc.function.arguments = '{"pattern": "test", "path": "/tmp"}'
            tcs.append(tc)
        assert _should_parallelize_tool_batch(tcs)


class TestTrajectoryNormalizeMsg:
    def test_import(self):
        from agent.utils import _trajectory_normalize_msg
        assert callable(_trajectory_normalize_msg)

    def test_strips_internal_fields(self):
        from agent.utils import _trajectory_normalize_msg
        # _trajectory_normalize_msg replaces image blobs with text placeholders
        msg = {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}
        result = _trajectory_normalize_msg(msg)
        # Image URL replaced with text placeholder
        assert result["content"][0]["type"] == "text"
        assert "screenshot" in result["content"][0]["text"].lower()

    def test_preserves_role(self):
        from agent.utils import _trajectory_normalize_msg
        msg = {"role": "user", "content": "hello"}
        result = _trajectory_normalize_msg(msg)
        assert result["role"] == "user"


class TestIterationBudget:
    def test_import(self):
        from agent.utils import IterationBudget
        assert callable(IterationBudget)

    def test_consume_decrements_remaining(self):
        from agent.utils import IterationBudget
        budget = IterationBudget(max_total=10)
        assert budget.remaining == 10
        budget.consume()
        assert budget.remaining == 9

    def test_consume_returns_false_when_exhausted(self):
        from agent.utils import IterationBudget
        budget = IterationBudget(max_total=2)
        budget.consume()
        budget.consume()
        assert not budget.consume()

    def test_refund(self):
        from agent.utils import IterationBudget
        budget = IterationBudget(max_total=5)
        budget.consume()
        budget.refund()
        assert budget.remaining == 5

    def test_used_property(self):
        from agent.utils import IterationBudget
        budget = IterationBudget(max_total=5)
        budget.consume()
        budget.consume()
        assert budget.used == 2
