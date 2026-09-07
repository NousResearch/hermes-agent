"""Real SQLite/iteration proof for rejected Astra steering, with and without tools."""

import threading
from types import SimpleNamespace

import pytest

from tests.agent.test_astra_websocket import FakeWebSocket, _connect_socket, _created, _completed


@pytest.mark.parametrize("with_tool", [False, True])
@pytest.mark.parametrize("crash", [None, "rollback", "after_commit"])
def test_failed_steer_redirect_is_atomic_and_restart_safe(monkeypatch, tmp_path, with_tool, crash):
    from agent import turn_iteration_prep
    from agent.transports.astra_websocket_session import AstraSteeringPersistenceError, AstraWebSocketSession
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *args, **kwargs: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])
    db = SessionDB(tmp_path / "state.db")
    sid, agents = "synthetic-rejected-steer", []

    def make_agent():
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
        messages = db.get_messages_as_conversation(sid, include_row_ids=True)
        return turn_iteration_prep.begin_iteration(
            agent, messages=messages, conversation_history=messages[:], original_user_message="repeat me",
            api_call_count=0, interrupted=False, _turn_exit_reason=None,
        )

    first = make_agent()
    # Identical earlier user text must not consume the later admission by content equality.
    messages = [{"role": "user", "content": "repeat me"}]
    if with_tool:
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "original-call", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "saved tool result", "tool_call_id": "original-call"},
        ]
    messages.append({"role": "assistant", "content": "completed answer"})
    assert first._flush_messages_to_session_db(messages) is True
    first._session_messages = messages
    socket, active, errors = FakeWebSocket(), threading.Event(), []
    session = AstraWebSocketSession(first, connect=_connect_socket(socket))

    def on_send(message, ws):
        if message["type"] == "response.create":
            ws.push(_created("r1"))
        elif message["type"] == "response.steer":
            ws.push({"type": "response.steer.failed", "id": "failed"})
            for event in _completed("r1", "r1"):
                ws.push(event)

    socket._on_send = on_send
    original_created = session._handle_created

    def created(event, response_id):
        original_created(event, response_id)
        active.set()

    session._handle_created = created

    def run():
        try:
            session.run({"model": "gpt-6-astra", "input": [{"role": "user", "content": "repeat me"}]})
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert active.wait(2)
        assert session.request_steer("repeat me") is True
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert first._pending_steer is None  # Never the external next-user string handoff.
        assert first._pending_redirect == "repeat me"
        before = db.get_messages(sid)

        if crash == "rollback":
            original_ack = db._ack_astra_fallback_redirects

            def fail_ack(*args):
                original_ack(*args)
                raise OSError("synthetic transaction rollback")

            with monkeypatch.context() as patch:
                patch.setattr(db, "_ack_astra_fallback_redirects", fail_ack)
                with pytest.raises(AstraSteeringPersistenceError):
                    begin(make_agent())
            assert db.get_messages(sid) == before
            assert db.get_session_model_config_value(sid, "_astra_steering")["entries"][-1]["state"] == "failed"
        elif crash == "after_commit":
            def stop_after_commit(*args):
                raise RuntimeError("synthetic process loss after commit")

            with monkeypatch.context() as patch:
                patch.setattr("agent.transports.astra_websocket_session.confirm_astra_redirect_persisted", stop_after_commit)
                with pytest.raises(RuntimeError, match="process loss"):
                    begin(make_agent())

        assert begin(make_agent()).action == "fallthrough"
        after = db.get_messages(sid)
        assert [row["content"] for row in after if row["role"] == "user"] == ["repeat me", "repeat me"]
        receipt = db.get_session_model_config_value(sid, "_astra_steering")["entries"][-1]
        assert receipt["state"] == "fallback_delivered"
        delivered = next(row for row in after if row["id"] == receipt["fallback_message_row_id"])
        assert delivered["display_metadata"]["_astra_steering_fallback"][0]["admission_id"] == receipt["admission_id"]
        assert [row["content"] for row in after if row["role"] == "tool"] == (["saved tool result"] if with_tool else [])
        assert begin(make_agent()).action == "fallthrough"
        assert db.get_messages(sid) == after
        assert sum(item["type"] == "response.steer" for item in socket.sent) == 1
    finally:
        session.close()
        worker.join(2)
        for agent in agents:
            agent.close()
        db.close()
