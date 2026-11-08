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


from gateway.run import _preserve_queued_followup_history_offset


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
        if role in {"session_meta",}:
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


    def test_recursive_queued_followup_keeps_outer_history_offset(self):
        """Queued drain persistence must include every turn in the chain.

        ``_run_agent()`` recurses when a follow-up arrived while the current turn
        was running. The recursive call naturally returns a later
        ``history_offset`` because it received the previous turn as part of its
        input history. If the outer caller persists transcript rows using that
        later offset, it only sees the *last* queued turn as new and drops the
        earlier queued turn from the transcript.
        """
        history_before_chain = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        first_followup_turn = [
            {"role": "user", "content": "First follow-up question"},
            {"role": "assistant", "content": "First follow-up answer"},
        ]
        second_followup_turn = [
            {"role": "user", "content": "Second follow-up question"},
            {"role": "assistant", "content": "Second follow-up answer"},
        ]

        current_result = {
            "history_offset": len(history_before_chain),
            "messages": history_before_chain + first_followup_turn,
        }
        followup_result = {
            "history_offset": len(history_before_chain + first_followup_turn),
            "messages": (
                history_before_chain
                + first_followup_turn
                + second_followup_turn
            ),
        }

        merged = _preserve_queued_followup_history_offset(
            current_result,
            followup_result,
        )
        assert merged["history_offset"] == len(history_before_chain)

        persisted = merged["messages"][merged["history_offset"]:]
        assert persisted == first_followup_turn + second_followup_turn

