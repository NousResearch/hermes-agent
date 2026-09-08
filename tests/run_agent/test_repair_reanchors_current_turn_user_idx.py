"""Regression: ``repair_message_sequence`` merges adjacent rows *before* the current
turn's user message (a role=user compaction summary next to the protected first
user message), so the index recorded at turn start drifts past the current row.
Hosts that settle the transcript by that index (WebUI) then write the current
user turn to the FRONT of the context. ``run_conversation`` must re-anchor the
index after a repair that changed the list.
"""
from agent.agent_runtime_helpers import repair_message_sequence
from agent.turn_context import reanchor_current_turn_user_idx


class _Agent:
    session_id = "s"


def _history_with_adjacent_users():
    return [
        {"role": "assistant", "content": "**Context snapshot**"},
        {"role": "user", "content": "compaction summary written as a user row"},
        {"role": "user", "content": "first protected user message"},
        {"role": "assistant", "content": "ok",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
        {"role": "user", "content": "NEW question"},
    ]


def test_repair_shifts_recorded_index_and_reanchor_recovers_it():
    messages = _history_with_adjacent_users()
    recorded_idx = len(messages) - 1  # what run_conversation records at turn start
    assert messages[recorded_idx]["content"] == "NEW question"
    repairs = repair_message_sequence(_Agent(), messages)
    assert repairs >= 1
    # the recorded index no longer addresses the current user row
    assert recorded_idx >= len(messages) or messages[recorded_idx]["content"] != "NEW question"
    reanchored = reanchor_current_turn_user_idx(messages, "NEW question")
    assert messages[reanchored]["role"] == "user"
    assert messages[reanchored]["content"] == "NEW question"
    assert reanchored == len(messages) - 1


def test_repair_without_changes_keeps_index():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "NEW question"},
    ]
    recorded_idx = len(messages) - 1
    assert repair_message_sequence(_Agent(), messages) == 0
    assert reanchor_current_turn_user_idx(messages, "NEW question") == recorded_idx
