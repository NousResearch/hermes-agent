"""Contract tests for the cron memory write guard (#45769).

The guard must treat BOTH documented ``memory()`` call shapes as writes:
the single-op ``action`` form and the batch ``operations`` form (whose
entries carry their own action field). Before this fix the dispatch-site
guard only checked top-level ``action``, so a batch write bypassed it.

``memory_call_is_write`` is a pure predicate shared by the sequential
(``agent/tool_executor.py``) and concurrent (``agent/agent_runtime_helpers.py``)
dispatch sites — testing the predicate directly covers both.
"""
import pytest

from agent.agent_runtime_helpers import memory_call_is_write


class TestMemoryCallIsWrite:
    def test_single_op_add_is_write(self):
        assert memory_call_is_write("add", None) is True

    def test_single_op_replace_is_write(self):
        assert memory_call_is_write("replace", None) is True

    def test_single_op_remove_is_write(self):
        assert memory_call_is_write("remove", None) is True

    def test_single_op_read_is_not_write(self):
        assert memory_call_is_write("recall", None) is False

    def test_none_action_without_operations_is_not_write(self):
        # The batch shape omits top-level action entirely.
        assert memory_call_is_write(None, None) is False

    def test_batch_add_op_is_write(self):
        ops = [{"action": "add", "content": "x"}]
        assert memory_call_is_write(None, ops) is True

    def test_batch_replace_op_is_write(self):
        ops = [{"action": "replace", "old_text": "x", "content": "y"}]
        assert memory_call_is_write(None, ops) is True

    def test_batch_remove_op_is_write(self):
        ops = [{"action": "remove", "old_text": "x"}]
        assert memory_call_is_write(None, ops) is True

    def test_batch_read_ops_are_not_write(self):
        ops = [{"action": "recall", "query": "x"}, {"action": "list"}]
        assert memory_call_is_write(None, ops) is False

    def test_mixed_batch_with_any_write_op_is_write(self):
        ops = [{"action": "recall", "query": "x"}, {"action": "add", "content": "y"}]
        assert memory_call_is_write(None, ops) is True

    def test_empty_operations_is_not_write(self):
        assert memory_call_is_write(None, []) is False

    def test_non_list_operations_is_not_write(self):
        assert memory_call_is_write(None, {"action": "add"}) is False

    def test_malformed_entries_are_ignored(self):
        ops = ["not-a-dict", {"action": "recall"}, None]
        assert memory_call_is_write(None, ops) is False

    def test_top_level_action_beats_empty_operations(self):
        # Even with an empty operations list, an explicit top-level write
        # action must still be caught.
        assert memory_call_is_write("add", []) is True
