"""Alternation repair must invalidate persistence markers it mutates past.

Merging/pruning rewrites flushed dicts in place. A surviving row that keeps
its ``_db_persisted`` marker makes the next flush skip it, leaving the DB
with pre-merge bytes while the live transcript (and resume, and prompt
cache) moved on. Same contract as the api_content sidecar the merges
already drop.
"""

from types import SimpleNamespace

from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import _DB_PERSISTED_MARKER


def _agent():
    return SimpleNamespace(_db_flush_scan_prefix=[{"sentinel": True}])


class TestRepairPersistenceInvalidation:
    def test_user_merge_pops_marker_and_prefix(self):
        agent = _agent()
        messages = [
            {"role": "user", "content": "first", _DB_PERSISTED_MARKER: True},
            {"role": "user", "content": "second"},
        ]
        assert repair_message_sequence(agent, messages) == 1
        assert len(messages) == 1
        assert messages[0]["content"] == "first\n\nsecond"
        assert _DB_PERSISTED_MARKER not in messages[0]
        assert agent._db_flush_scan_prefix is None

    def test_assistant_merge_pops_marker(self):
        agent = _agent()
        messages = [
            {"role": "assistant", "content": "part one", _DB_PERSISTED_MARKER: True},
            {"role": "assistant", "content": "part two"},
        ]
        assert repair_message_sequence(agent, messages) == 1
        assert len(messages) == 1
        assert messages[0]["content"] == "part one\npart two"
        assert _DB_PERSISTED_MARKER not in messages[0]

    def test_no_repair_leaves_markers_and_prefix_alone(self):
        agent = _agent()
        prefix = agent._db_flush_scan_prefix
        messages = [
            {"role": "user", "content": "task", _DB_PERSISTED_MARKER: True},
            {"role": "assistant", "content": "done"},
        ]
        assert repair_message_sequence(agent, messages) == 0
        assert messages[0][_DB_PERSISTED_MARKER] is True
        assert agent._db_flush_scan_prefix is prefix

    def test_agent_without_prefix_attr_still_repairs(self):
        agent = SimpleNamespace()
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        assert repair_message_sequence(agent, messages) == 1
        assert messages[0]["content"] == "first\n\nsecond"
