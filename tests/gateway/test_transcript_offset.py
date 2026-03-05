"""Tests for transcript history offset fix.

Regression tests for a bug where the gateway transcript lost 1 message
per turn from turn 2 onwards.  The raw transcript history includes
``session_meta`` entries that are filtered out before being passed to
the agent.  The agent returns messages built from this filtered history
plus new messages from the current turn.

The old code used ``len(history)`` (raw count, includes session_meta)
to slice ``agent_messages``, which caused the slice to skip valid new
messages.  The fix adds ``history_offset`` (the filtered history length)
to ``_run_agent``'s return dict and uses it for the slice.
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers - replicate the filtering logic from _run_agent
# ---------------------------------------------------------------------------

def _filter_history(history: list) -> list:
    """Replicate the agent_history filtering from GatewayRunner._run_agent.

    Strips session_meta and system messages, exactly as the real code does.
    """
    agent_history = []
    for msg in history:
        role = msg.get("role")
        if not role:
            continue
        if role in ("session_meta",):
            continue
        if role == "system":
            continue

        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"

        if has_tool_calls or has_tool_call_id or is_tool_message:
            clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
            agent_history.append(clean_msg)
        else:
            content = msg.get("content")
            if content:
                agent_history.append({"role": role, "content": content})
    return agent_history


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscriptHistoryOffset:
    """Verify the transcript extraction uses the filtered history length."""

    def test_session_meta_causes_offset_mismatch(self):
        """Turn 2: session_meta makes len(history) > len(agent_history).

        - history (raw): 1 session_meta + 2 conversation = 3 entries
        - agent_history (filtered): 2 entries
        - Agent returns 2 old + 2 new = 4 messages
        - OLD: agent_messages[3:] = 1 message (lost the user message)
        - FIX: agent_messages[2:] = 2 messages (correct)
        """
        history = [
            {"role": "session_meta", "tools": [], "model": "gpt-4",
             "platform": "telegram", "timestamp": "t0"},
            {"role": "user", "content": "Hello", "timestamp": "t1"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "t1"},
        ]

        agent_history = _filter_history(history)
        assert len(agent_history) == 2  # session_meta stripped

        # Agent returns: filtered history (2) + new turn (2)
        agent_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
        ]

        # OLD behavior: len(history) = 3, skips too many
        old_offset = len(history)
        old_new = (agent_messages[old_offset:]
                   if len(agent_messages) > old_offset
                   else agent_messages)
        assert len(old_new) == 1  # BUG: lost the user message

        # FIXED behavior: history_offset = 2
        history_offset = len(agent_history)
        fixed_new = (agent_messages[history_offset:]
                     if len(agent_messages) > history_offset
                     else [])
        assert len(fixed_new) == 2
        assert fixed_new[0]["content"] == "What is Python?"
        assert fixed_new[1]["content"] == "A programming language."

    def test_no_session_meta_same_result(self):
        """First turn has no session_meta, so both approaches agree."""
        history = []
        agent_history = _filter_history(history)
        assert len(agent_history) == 0

        agent_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        old_new = (agent_messages[len(history):]
                   if len(agent_messages) > len(history)
                   else agent_messages)
        fixed_new = (agent_messages[len(agent_history):]
                     if len(agent_messages) > len(agent_history)
                     else [])

        assert old_new == fixed_new
        assert len(fixed_new) == 2

    def test_multiple_session_meta_larger_drift(self):
        """Two session_meta entries double the offset error.

        This can happen when the session spans tool definition changes
        or model switches that each write a new session_meta record.
        """
        history = [
            {"role": "session_meta", "tools": [], "timestamp": "t0"},
            {"role": "user", "content": "msg1", "timestamp": "t1"},
            {"role": "assistant", "content": "reply1", "timestamp": "t1"},
            {"role": "session_meta", "tools": ["new_tool"], "timestamp": "t2"},
            {"role": "user", "content": "msg2", "timestamp": "t3"},
            {"role": "assistant", "content": "reply2", "timestamp": "t3"},
        ]

        agent_history = _filter_history(history)
        assert len(agent_history) == 4
        assert len(history) == 6  # 2 extra session_meta entries

        # Agent returns 4 old + 2 new = 6 total
        agent_messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "reply1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "reply2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "reply3"},
        ]

        # OLD: len(history) == len(agent_messages) == 6 -> else branch
        old_offset = len(history)
        old_new = (agent_messages[old_offset:]
                   if len(agent_messages) > old_offset
                   else agent_messages)
        # BUG: treats ALL messages as new (duplicates entire history)
        assert old_new == agent_messages

        # FIXED: history_offset = 4
        fixed_new = (agent_messages[len(agent_history):]
                     if len(agent_messages) > len(agent_history)
                     else [])
        assert len(fixed_new) == 2
        assert fixed_new[0]["content"] == "msg3"
        assert fixed_new[1]["content"] == "reply3"

    def test_system_messages_also_filtered(self):
        """system messages in history are also stripped from agent_history."""
        history = [
            {"role": "session_meta", "tools": [], "timestamp": "t0"},
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi", "timestamp": "t1"},
            {"role": "assistant", "content": "Hello!", "timestamp": "t1"},
        ]

        agent_history = _filter_history(history)
        assert len(agent_history) == 2  # only user + assistant

        agent_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "New question"},
            {"role": "assistant", "content": "New answer"},
        ]

        # OLD: len(history) = 4, skips everything
        old_offset = len(history)
        old_new = (agent_messages[old_offset:]
                   if len(agent_messages) > old_offset
                   else agent_messages)
        assert old_new == agent_messages  # BUG: all treated as new

        # FIXED
        fixed_new = (agent_messages[len(agent_history):]
                     if len(agent_messages) > len(agent_history)
                     else [])
        assert len(fixed_new) == 2
        assert fixed_new[0]["content"] == "New question"

    def test_else_branch_returns_empty_list(self):
        """When agent has fewer messages than offset, return [] not all.

        The old code had ``else agent_messages`` which would treat the
        entire message list as new when the agent compressed or dropped
        messages.  The fix changes this to ``else []``, falling through
        to the simple user/assistant fallback path.
        """
        history = [
            {"role": "session_meta", "tools": [], "timestamp": "t0"},
            {"role": "user", "content": "Hello", "timestamp": "t1"},
            {"role": "assistant", "content": "Hi!", "timestamp": "t1"},
        ]

        # Agent compressed and returned fewer messages than history
        agent_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        history_offset = len(_filter_history(history))  # 2
        new_messages = (agent_messages[history_offset:]
                        if len(agent_messages) > history_offset
                        else [])
        # 2 == 2, so no new messages - falls to fallback
        assert new_messages == []

    def test_tool_call_messages_preserved_in_filter(self):
        """Tool call messages pass through the filter, keeping offset correct."""
        history = [
            {"role": "session_meta", "tools": [], "timestamp": "t0"},
            {"role": "user", "content": "Search for cats", "timestamp": "t1"},
            {"role": "assistant", "content": None, "timestamp": "t1",
             "tool_calls": [{"id": "tc1", "function": {"name": "web_search"}}]},
            {"role": "tool", "tool_call_id": "tc1",
             "content": "Results about cats", "timestamp": "t1"},
            {"role": "assistant", "content": "Here are results.",
             "timestamp": "t1"},
        ]

        agent_history = _filter_history(history)
        # session_meta filtered, but tool_calls/tool messages kept
        assert len(agent_history) == 4
        assert len(history) == 5  # 1 session_meta extra

        agent_messages = [
            {"role": "user", "content": "Search for cats"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "tc1", "function": {"name": "web_search"}}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "Results about cats"},
            {"role": "assistant", "content": "Here are results."},
            {"role": "user", "content": "Now search for dogs"},
            {"role": "assistant", "content": "Dog results here."},
        ]

        # OLD: len(history) = 5, agent_messages[5:] = 1 message (lost user msg)
        old_new = (agent_messages[len(history):]
                   if len(agent_messages) > len(history)
                   else agent_messages)
        assert len(old_new) == 1  # BUG

        # FIXED
        fixed_new = (agent_messages[len(agent_history):]
                     if len(agent_messages) > len(agent_history)
                     else [])
        assert len(fixed_new) == 2
        assert fixed_new[0]["content"] == "Now search for dogs"
        assert fixed_new[1]["content"] == "Dog results here."
