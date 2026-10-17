"""``prepare_iteration`` runs the alternation repair, which merges adjacent user rows in place
(after a compaction the role=user summary sits next to the protected first user message). The
index recorded at turn start then points past this turn's user row; hosts that settle the
transcript by that index (WebUI) write the current turn to the FRONT of the context. The
iteration prep must hand back a re-anchored index and mirror it into the persist override.

The same pass also re-anchors an index left stale by mid-turn compaction, and that second path
must report only an index it actually moved: a merge-into-tail rewrite leaves the guard
unsatisfied while the remedy's fallback resolves back to the same row."""


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


def test_prepare_iteration_does_not_report_a_reanchor_that_did_not_move(tmp_path, monkeypatch, caplog):
    """The stale-index guard is driven by this turn's verbatim text, while the remedy falls back
    to the last user-originated row. A merge-into-tail rewrite satisfies neither of the guard's
    matches yet leaves the fallback on the same row, so old == new, the branch re-enters on every
    iteration, and the diagnostic used to claim a re-anchor that never happened.

    Here the recorded index already addresses this turn's row, so nothing needs fixing and
    nothing may be reported as fixed."""

    import logging

    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import _anchors_current_turn, prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        _reset_per_turn_agent_state(agent)
        user_message = "the ask for this turn"
        messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "answer"},
            # merge-summary-into-tail rewrote this row: the verbatim ask is nowhere in the list
            {"role": "user", "content": "[merged compaction summary] + tail of the ask"},
        ]
        recorded_idx = 2
        agent._persist_user_message_idx = recorded_idx

        # precondition: the guard is unsatisfiable here and the remedy is a no-op
        assert _anchors_current_turn(messages, recorded_idx, user_message) is False

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            prep = prepare_iteration(
                agent, messages=messages, api_call_count=1,
                user_message=user_message, current_turn_user_idx=recorded_idx,
            )

        assert prep.action == "fallthrough"
        assert prep.current_turn_user_idx == recorded_idx
        assert agent._persist_user_message_idx == recorded_idx
        assert not any(
            "Re-anchored stale current_turn_user_idx" in r.getMessage() for r in caplog.records
        )
    finally:
        agent._session_db.close()


def test_prepare_iteration_still_reports_a_reanchor_that_moved_the_index(tmp_path, monkeypatch, caplog):
    """The diagnostic must keep firing whenever the recorded index really is stale — for example
    when a mid-turn compaction shrank the transcript so the coordinate now points past this
    turn's row."""

    import logging

    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import _anchors_current_turn, prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        _reset_per_turn_agent_state(agent)
        user_message = "the ask for this turn"
        messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": user_message},
        ]
        stale_idx = 5  # recorded before a compaction shrank the list to three rows

        assert _anchors_current_turn(messages, stale_idx, user_message) is False

        with caplog.at_level(logging.INFO, logger="agent.conversation_loop"):
            prep = prepare_iteration(
                agent, messages=messages, api_call_count=1,
                user_message=user_message, current_turn_user_idx=stale_idx,
            )

        assert prep.current_turn_user_idx == 2 != stale_idx
        assert agent._persist_user_message_idx == 2
        assert any(
            "Re-anchored stale current_turn_user_idx 5 -> 2" in r.getMessage()
            for r in caplog.records
        )
    finally:
        agent._session_db.close()
