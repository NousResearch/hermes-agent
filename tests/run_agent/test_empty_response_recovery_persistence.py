"""Regression tests for empty-response recovery transcript persistence."""

from run_agent import AIAgent


def _agent_with_stubbed_persistence():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._session_db = None
    agent._session_messages = []
    agent.saved_session_logs = []
    agent.flushed_session_db_messages = []
    agent._save_session_log = lambda messages: agent.saved_session_logs.append(
        [m.copy() for m in messages]
    )
    agent._flush_messages_to_session_db = lambda messages, conversation_history=None: (
        agent.flushed_session_db_messages.append([m.copy() for m in messages])
    )
    return agent


def test_persist_session_strips_trailing_empty_recovery_scaffolding():
    agent = _agent_with_stubbed_persistence()
    messages = [
        {"role": "user", "content": "run the task"},
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": (
                "You just executed tool calls but returned an empty response. "
                "Please process the tool results above and continue with the task."
            ),
            "_empty_recovery_synthetic": True,
        },
    ]

    AIAgent._persist_session(agent, messages, conversation_history=[])

    assert messages == [
        {"role": "user", "content": "run the task"},
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
    ]
    assert agent.saved_session_logs[-1] == messages
    assert all(not msg.get("_empty_recovery_synthetic") for msg in messages)


def test_persist_session_keeps_real_terminal_empty_response():
    agent = _agent_with_stubbed_persistence()
    messages = [
        {"role": "user", "content": "run the task"},
        {"role": "assistant", "content": "(empty)"},
    ]

    AIAgent._persist_session(agent, messages, conversation_history=[])

    assert messages == [
        {"role": "user", "content": "run the task"},
        {"role": "assistant", "content": "(empty)"},
    ]
    assert agent.saved_session_logs[-1] == messages
