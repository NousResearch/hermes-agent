"""Regression for #84235: the API server reads a session's history before the turn takes the
session lease, so a turn another writer finishes in between must still reach the model."""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hermes_state import SessionDB
from tests.agent.test_cross_process_turn_lease import _agent_with_db
from tests.gateway.test_api_server import _create_app, _make_adapter


@pytest.mark.asyncio
async def test_turn_finished_between_history_load_and_admission_reaches_the_model(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    other_writer, served = SessionDB(path), SessionDB(path)
    other_writer.create_session("shared", source="cli")
    other_writer.append_message("shared", "user", "first")
    observed = {}

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        observed["history"] = [m.get("content") for m in history]
        return {"final_response": "ok", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_run)
    adapter = _make_adapter(api_key="sk-secret")
    adapter._session_db = served

    def create_agent(**_kwargs):
        # Runs after the handler read the history and before the turn asks for the lease: the
        # other writer's turn ends here, so the lease is free and nothing makes this turn wait.
        other_writer.append_message("shared", "assistant", "first answer")
        agent = _agent_with_db(served, session_id="shared", platform="api_server")
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        return agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    try:
        async with TestClient(TestServer(_create_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Session-Id": "shared", "Authorization": "Bearer sk-secret"},
                json={"model": "hermes-agent", "messages": [{"role": "user", "content": "second"}]})
            assert resp.status == 200, await resp.text()
    finally:
        other_writer.close()
        served.close()

    assert observed["history"] == ["first", "first answer"]
