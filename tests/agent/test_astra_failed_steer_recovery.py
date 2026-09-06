"""Same-agent retry and rotating local compression use durable steering truth."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def rejected_steer(monkeypatch, tmp_path):
    from agent import turn_iteration_prep
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *args, **kwargs: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])
    db, agents = SessionDB(tmp_path / "state.db"), []

    def make_agent(sid="synthetic-retry-parent"):
        agent = AIAgent(
            model="gpt-6-astra", provider="openai", api_key="placeholder-key",
            base_url="https://api.openai.com/v1", platform="cli", session_id=sid, session_db=db,
            quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True,
        )
        agent._checkpoint_mgr = SimpleNamespace(new_turn=lambda: None)
        agent._budget_grace_call = False
        agent.iteration_budget = SimpleNamespace(consume=lambda: True)
        agents.append(agent)
        return agent

    def begin(agent):
        # The CLI keeps the agent but resets turn messages after an exception.
        messages = db.get_messages_as_conversation(agent.session_id, include_row_ids=True)
        return turn_iteration_prep.begin_iteration(
            agent, messages=messages, conversation_history=messages[:], original_user_message="repeat me",
            api_call_count=0, interrupted=False, _turn_exit_reason=None,
        )

    first = make_agent()
    assert first._flush_messages_to_session_db([
        {"role": "user", "content": "repeat me"},
        {"role": "assistant", "content": "completed answer"},
    ]) is True
    # The WebSocket/replay suite proves production admission -> explicit failure.
    # Start this recovery gate at that durable state, without any provider call.
    db.patch_session_model_config(first.session_id, {"_astra_steering": {"entries": [{
        "admission_id": "synthetic-rejected-admission", "input_sha256": "synthetic-digest",
        "text": "repeat me", "state": "failed",
    }]}})
    try:
        yield db, first, make_agent, begin
    finally:
        for agent in agents:
            agent.close()
        db.close()


@pytest.mark.parametrize("failure", ["rollback", "readback", "persist_raises"])
def test_same_agent_retry_reloads_journal_without_duplicate_delivery(monkeypatch, rejected_steer, failure):
    from agent.transports.astra_websocket_session import AstraSteeringPersistenceError

    db, agent, _, begin = rejected_steer
    before = db.get_messages(agent.session_id)
    with monkeypatch.context() as patch:
        if failure == "rollback":
            original_ack = db._ack_astra_fallback_redirects

            def fail_ack(*args):
                original_ack(*args)
                raise OSError("synthetic rollback after acknowledgement")

            patch.setattr(db, "_ack_astra_fallback_redirects", fail_ack)
            expected = AstraSteeringPersistenceError
        elif failure == "readback":
            original_get = db.get_session_model_config_value

            def fail_readback(*args, **kwargs):
                value = original_get(*args, **kwargs)
                if args[1] == "_astra_steering" and value["entries"][-1]["state"] == "fallback_delivered":
                    raise OSError("synthetic readback failure after commit")
                return value

            patch.setattr(db, "get_session_model_config_value", fail_readback)
            expected = OSError
        else:
            def fail_persist(*args):
                raise OSError("synthetic persistence exception before commit")

            patch.setattr(agent, "_persist_session", fail_persist)
            expected = OSError
        with pytest.raises(expected):
            begin(agent)
    if failure != "readback":
        assert db.get_messages(agent.session_id) == before
    assert agent._astra_fallback_restore_session is None
    assert not agent._astra_drained_redirect_receipts
    assert begin(agent).action == "fallthrough"
    after = db.get_messages(agent.session_id)
    assert [row["content"] for row in after if row["role"] == "user"] == ["repeat me", "repeat me"]
    receipt = db.get_session_model_config_value(agent.session_id, "_astra_steering")["entries"][-1]
    assert receipt["state"] == "fallback_delivered"
    assert receipt["fallback_message_row_id"] == after[-1]["id"]
    assert begin(agent).action == "fallthrough"
    assert db.get_messages(agent.session_id) == after


@pytest.mark.parametrize("delivered", [False, True])
def test_restart_rotation_carries_current_journal_atomically(monkeypatch, rejected_steer, delivered):
    from agent.conversation_compression import _publish_rotated_compaction

    db, first, make_agent, begin = rejected_steer
    if delivered:
        assert begin(first).action == "fallthrough"
    parent_sid = first.session_id
    journal = db.get_session_model_config_value(parent_sid, "_astra_steering")
    restarted = make_agent(parent_sid)
    assert "_astra_steering" not in (restarted._session_init_model_config or {})
    history = db.get_messages_as_conversation(parent_sid, include_row_ids=True)
    handoff = [{"role": "user", "content": "retained task facts"},
               {"role": "assistant", "content": "local compression handoff"}]
    original_publish, publications = db._publish_child_session_row, []

    def inspect_publication(conn, parent, **kwargs):
        assert conn.in_transaction
        assert kwargs["model_config"]["_astra_steering"] == journal
        publications.append(kwargs["child_session_id"])
        return original_publish(conn, parent, **kwargs)

    monkeypatch.setattr(db, "_publish_child_session_row", inspect_publication)
    _publish_rotated_compaction(
        restarted, history, handoff, new_system_prompt="same system prompt\n",
        lease=SimpleNamespace(holder=None, ttl=300, watermark=None), old_session_id=parent_sid,
        compressed_user_turn_outcome="already_present",
    )
    child_sid = restarted.session_id
    assert publications == [child_sid]
    assert db.find_live_compression_child(parent_sid)["id"] == child_sid
    assert db.get_session(parent_sid)["end_reason"] == "compression"
    assert db.get_session_model_config_value(child_sid, "_astra_steering") == journal
    after_rotation = make_agent(child_sid)
    assert begin(after_rotation).action == "fallthrough"
    after = db.get_messages(child_sid)
    assert [row["content"] for row in after if row["role"] == "user"] == (
        ["retained task facts"] if delivered else ["retained task facts", "repeat me"]
    )
    assert db.get_session_model_config_value(child_sid, "_astra_steering")["entries"][-1]["state"] == "fallback_delivered"
    assert begin(make_agent(child_sid)).action == "fallthrough"
    assert db.get_messages(child_sid) == after
