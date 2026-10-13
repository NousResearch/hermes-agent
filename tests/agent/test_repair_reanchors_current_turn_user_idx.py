"""``prepare_iteration`` runs the alternation repair, which merges adjacent user rows in place
(after a compaction the role=user summary sits next to the protected first user message). The
index recorded at turn start then points past this turn's user row; hosts that settle the
transcript by that index (WebUI) write the current turn to the FRONT of the context. The
iteration prep must hand back a re-anchored index and mirror it into the persist override."""


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from hermes_state import SessionDB

    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def test_prepare_iteration_reanchors_after_the_repair_merges_rows(tmp_path, monkeypatch):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        _reset_per_turn_agent_state(agent)
        messages = [
            {"role": "assistant", "content": "**Context snapshot**"},
            {"role": "user", "content": "compaction summary written as a user row"},
            {"role": "user", "content": "first protected user message"},
            {"role": "assistant", "content": "ok",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
            {"role": "user", "content": "NEW question"},
        ]
        recorded_idx = len(messages) - 1  # what run_conversation records at turn start
        prep = prepare_iteration(
            agent, messages=messages, api_call_count=1,
            user_message="NEW question", current_turn_user_idx=recorded_idx,
        )
        assert prep.action == "fallthrough"
        assert len(prep.messages) < len(messages) + 1 and recorded_idx >= len(prep.messages)
        assert prep.messages[prep.current_turn_user_idx]["content"] == "NEW question"
        assert agent._persist_user_message_idx == prep.current_turn_user_idx
    finally:
        agent._session_db.close()
