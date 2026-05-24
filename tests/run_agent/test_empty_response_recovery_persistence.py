"""Regression tests for empty-response recovery transcript persistence."""

from run_agent import AIAgent


def _agent_with_stubbed_persistence():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._session_db = None
    agent._session_messages = []
    agent._last_flushed_db_idx = 0
    agent.flushed_session_db_messages = []
    agent._flush_messages_to_session_db = lambda messages, conversation_history=None: (
        agent.flushed_session_db_messages.append([m.copy() for m in messages])
    )
    return agent


def test_persist_session_strips_trailing_empty_recovery_scaffolding():
    """After stripping scaffolding, also rewind past orphan trailing tool-result
    messages that the failed iteration left behind. Otherwise the next user
    message lands after a bare ``tool`` and produces a protocol-invalid
    sequence that most providers silently fail on, retriggering the empty-
    retry loop indefinitely.
    """
    agent = _agent_with_stubbed_persistence()
    messages = [
        {"role": "user", "content": "run the task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
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

    # After strip + rewind, only the original user message remains. The
    # assistant(tool_calls) + tool pair is dropped because its iteration
    # never produced a real response.
    assert messages == [
        {"role": "user", "content": "run the task"},
    ]
    assert agent.flushed_session_db_messages[-1] == messages
    assert all(not msg.get("_empty_recovery_synthetic") for msg in messages)


def test_persist_session_keeps_unmarked_terminal_empty_response():
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
    assert agent.flushed_session_db_messages[-1] == messages


def test_persist_session_strips_marked_terminal_empty_sentinel():
    agent = _agent_with_stubbed_persistence()
    messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_terminal_sentinel": True,
        },
    ]

    AIAgent._persist_session(agent, messages, conversation_history=[])

    assert messages == [{"role": "user", "content": "continue"}]
    assert agent.flushed_session_db_messages[-1] == messages
    assert all(not msg.get("_empty_terminal_sentinel") for msg in messages)


def test_persist_session_clamps_flush_idx_after_scaffolding_drop():
    """Regression for #31507.

    An intermediate _persist_session that flushed while scaffolding was still
    present in ``messages`` leaves ``_last_flushed_db_idx`` at the pre-drop
    length. A later call drops the scaffolding (shrinking the list) and then
    must still flush newly-appended messages — without the clamp,
    ``messages[flush_from:]`` is empty and the final assistant response is
    silently lost.
    """
    agent = _agent_with_stubbed_persistence()

    # Simulate the previous flush that wrote the user + scaffolding tail
    # (4 messages total). The flush index was advanced to that length.
    agent._last_flushed_db_idx = 4

    # After the scaffolding got dropped and a new assistant turn was appended,
    # the list is now shorter than _last_flushed_db_idx.
    messages = [
        {"role": "user", "content": "run the task"},
        {"role": "assistant", "content": "final response"},
    ]

    AIAgent._persist_session(agent, messages, conversation_history=[])

    # The clamp must have happened so the next flush observes the real length.
    assert agent._last_flushed_db_idx <= len(messages)
    # And the final assistant must have reached the flush path.
    assert agent.flushed_session_db_messages[-1] == messages


def test_persist_session_flushes_new_assistant_after_scaffolding_drop_integration():
    """End-to-end regression for #31507 using a real SessionDB.

    Mirrors the production trigger: an intermediate flush writes a scaffolded
    tail; a later _persist_session drops the scaffolding and appends a real
    assistant response; that response must land in the DB.
    """
    import os
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "test.db")

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id="test-31507",
                skip_context_files=True,
                skip_memory=True,
            )
        agent._ensure_db_session()

        # Intermediate persist with scaffolding still in messages — gets flushed.
        messages = [
            {"role": "user", "content": "run the task"},
            {
                "role": "assistant",
                "content": "(empty)",
                "_empty_terminal_sentinel": True,
            },
        ]
        agent._flush_messages_to_session_db(messages, conversation_history=[])
        assert agent._last_flushed_db_idx == 2

        # A later turn appends the real assistant response. _persist_session
        # drops the sentinel then must still flush the new assistant.
        messages.append({"role": "assistant", "content": "real response"})
        agent._persist_session(messages, conversation_history=[])

        rows = db.get_messages(agent.session_id)
        contents = [r["content"] for r in rows]
        assert "real response" in contents, (
            f"final assistant lost (regression of #31507): got {contents}"
        )
