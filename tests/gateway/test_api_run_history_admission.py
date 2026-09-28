"""API runs load session-backed history inside admission, not when queued."""
from types import SimpleNamespace

import pytest

from gateway.platforms.api_server_runs import _run_history_kwargs
from hermes_state import SessionDB
from tests.agent.test_cross_process_turn_lease import _agent_with_db


@pytest.mark.parametrize("durable", [True, False])
@pytest.mark.parametrize("contended", [True, False])
def test_api_history_source_contract(tmp_path, monkeypatch, durable, contended):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("shared", source="api_server")
    db.append_message("shared", "user", "old input")
    db.append_message("shared", "assistant", "old reply")
    old = db.get_messages_as_conversation("shared", include_row_ids=True)
    agent = _agent_with_db(db, session_id="shared", platform="api_server")
    kw = _run_history_kwargs(SimpleNamespace(session_history_delivery=durable, conversation_history=old), agent)
    db.append_message("shared", "user", "last external input")
    db.append_message("shared", "assistant", "last external reply")
    if contended:
        assert db.try_acquire_session_turn_lease("shared", "holder")
        agent.status_callback = lambda kind, text=None: db.release_session_turn_lease("shared", "holder")
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 2)

    def loop(_agent, _message, _system, history, *args, **kwargs):
        if durable:
            assert history[-1]["content"] == "last external reply"
            assert not db.try_acquire_session_turn_lease("shared", "other")
        else:
            assert history is old
        return {"messages": history, "final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        assert agent.run_conversation("next", **kw)["final_response"] == "ok"
    finally:
        db.release_session_turn_lease("shared", "other")
        db.close()
