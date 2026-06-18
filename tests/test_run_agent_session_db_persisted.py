from types import SimpleNamespace

import run_agent


def test_run_conversation_reports_successful_session_db_persist(monkeypatch):
    fake_agent = SimpleNamespace(_session_db_persisted_this_turn=False)

    def _fake_run_conversation(agent, *args, **kwargs):
        agent._session_db_persisted_this_turn = True
        return {"final_response": "ok"}

    from agent import conversation_loop

    monkeypatch.setattr(conversation_loop, "run_conversation", _fake_run_conversation)
    result = run_agent.AIAgent.run_conversation(fake_agent, "hello")

    assert result["final_response"] == "ok"
    assert result["session_db_persisted"] is True


def test_run_conversation_reports_failed_session_db_persist(monkeypatch):
    fake_agent = SimpleNamespace(_session_db_persisted_this_turn=True)

    def _fake_run_conversation(agent, *args, **kwargs):
        agent._session_db_persisted_this_turn = False
        return {"final_response": "ok"}

    from agent import conversation_loop

    monkeypatch.setattr(conversation_loop, "run_conversation", _fake_run_conversation)
    result = run_agent.AIAgent.run_conversation(fake_agent, "hello")

    assert result["session_db_persisted"] is False
