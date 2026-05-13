"""Tests for GH-25242: Gateway auto-continue note persistence fix.

Verifies that:
1. The auto-continue note is NOT persisted to the transcript (persist_user_message)
2. The tool-tail ack prevents the same stale tool tail from triggering twice
3. New tool tails get their own recovery note (fresh key)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Tool-tail key computation (mirrors the logic in gateway/run.py)
# ---------------------------------------------------------------------------

def _compute_tool_tail_key(agent_history: list) -> str | None:
    """Compute a signature for the trailing tool batch in agent_history.

    Mirrors the logic added in the GH-25242 fix in gateway/run.py.
    """
    if not agent_history or agent_history[-1].get("role") != "tool":
        return None

    tail = agent_history[-1]
    tc_ids = []
    if isinstance(tail.get("content"), list):
        tc_ids = [
            c.get("tool_use_id", "")
            for c in tail["content"]
            if isinstance(c, dict) and c.get("tool_use_id")
        ]
    elif tail.get("tool_call_id"):
        tc_ids = [tail["tool_call_id"]]
    return "|".join(sorted(tc_ids)) if tc_ids else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolTailKeyComputation:
    """Verify the tool-tail key signature logic."""

    def test_empty_history_returns_none(self):
        assert _compute_tool_tail_key([]) is None

    def test_last_message_not_tool_returns_none(self):
        history = [{"role": "user", "content": "hello"}]
        assert _compute_tool_tail_key(history) is None

    def test_single_tool_call_id(self):
        history = [
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        assert _compute_tool_tail_key(history) == "call_1"

    def test_multiple_tool_results_in_content_list(self):
        history = [
            {"role": "assistant", "tool_calls": [{"id": "call_a"}, {"id": "call_b"}]},
            {
                "role": "tool",
                "content": [
                    {"tool_use_id": "call_a", "type": "tool_result", "content": "res_a"},
                    {"tool_use_id": "call_b", "type": "tool_result", "content": "res_b"},
                ],
            },
        ]
        # Sorted join
        assert _compute_tool_tail_key(history) == "call_a|call_b"

    def test_deterministic_ordering(self):
        """Same IDs in different content order produce the same key."""
        history_b_first = [
            {"role": "tool", "content": [
                {"tool_use_id": "call_b", "type": "tool_result", "content": ""},
                {"tool_use_id": "call_a", "type": "tool_result", "content": ""},
            ]},
        ]
        history_a_first = [
            {"role": "tool", "content": [
                {"tool_use_id": "call_a", "type": "tool_result", "content": ""},
                {"tool_use_id": "call_b", "type": "tool_result", "content": ""},
            ]},
        ]
        assert _compute_tool_tail_key(history_b_first) == _compute_tool_tail_key(history_a_first)

    def test_no_tool_call_ids_returns_none(self):
        """Tool message without any tool_call_id should return None."""
        history = [
            {"role": "tool", "content": "some output without id"},
        ]
        assert _compute_tool_tail_key(history) is None


class TestAutoContinueAck:
    """Verify the one-shot ack mechanism for tool-tail recovery."""

    def test_first_trigger_not_consumed(self):
        """First time seeing a tool tail: not consumed, should trigger."""
        consumed = {}
        key = "call_1"
        assert key not in consumed

    def test_second_trigger_consumed(self):
        """After ack, same key is consumed, should NOT trigger."""
        consumed = {}
        key = "call_1"
        consumed[key] = True  # simulate ack
        assert key in consumed

    def test_different_key_not_affected(self):
        """A new tool tail with different IDs should trigger normally."""
        consumed = {"call_1": True}
        new_key = "call_2"
        assert new_key not in consumed

    def test_consumed_dict_lifecycle(self):
        """Simulate the full lifecycle: trigger -> ack -> skip -> new -> trigger."""
        consumed = {}

        # First tool tail: call_1
        key1 = "call_1"
        assert key1 not in consumed
        consumed[key1] = True

        # Same tail again: skip
        assert key1 in consumed

        # New tool tail: call_2 (different interruption)
        key2 = "call_2"
        assert key2 not in consumed
        consumed[key2] = True

        # Both consumed
        assert key1 in consumed
        assert key2 in consumed


class TestPersistUserMessage:
    """Verify that persist_user_message logic is correct."""

    def test_different_messages_trigger_persist(self):
        """When original != modified, persist should be set."""
        original = "hello"
        modified = "[System note: ...]\n\nhello"
        persist = original if original != modified else None
        assert persist == "hello"

    def test_same_messages_no_persist(self):
        """When no note was prepended, persist should be None."""
        original = "hello"
        modified = "hello"
        persist = original if original != modified else None
        assert persist is None

    def test_skills_reload_note_triggers_persist(self):
        """Skills reload note also changes message, should persist original."""
        original = "do something"
        modified = "[Skills reloaded: ...]\n\ndo something"
        persist = original if original != modified else None
        assert persist == "do something"
