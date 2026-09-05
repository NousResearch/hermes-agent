"""Real executor and SQLite proof at the pending-tool steering boundary."""

import threading
from types import SimpleNamespace

import pytest

from tests.agent.test_astra_websocket import FakeWebSocket, _connect_socket, _created, _completed


@pytest.mark.parametrize("persistence_fails", [False, True])
@pytest.mark.parametrize("with_effort_update", [False, True])
def test_pending_steer_settles_real_executor_before_one_continuation(monkeypatch, tmp_path, persistence_fails, with_effort_update):
    from agent.astra_async_tools import AstraAsyncExecutor
    from agent import tool_executor
    from agent.transports.astra_websocket_session import AstraWebSocketSession, AstraProtocolError
    from hermes_state import SessionDB
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *args, **kwargs: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *args, **kwargs: [])
    db = SessionDB(tmp_path / "state.db")
    agent = AIAgent(
        model="gpt-6-astra", provider="openai", api_key="placeholder-key",
        base_url="https://api.openai.com/v1", platform="cli", session_id="synthetic-pending-steer",
        session_db=db, quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True,
    )
    messages = [{"role": "user", "content": "Use the read tool"}]
    if with_effort_update:
        messages = [
            {"role": "user", "content": "Start low", "_astra_reasoning_base_effort": "low"},
            {"role": "assistant", "content": "Ready"},
            {**messages[0], "_astra_configuration_update": {"type": "configuration_update", "reasoning": {"effort": "high"}}},
        ]
    assert agent._flush_messages_to_session_db(messages) is True
    request = agent._get_transport().build_kwargs(
        model=agent.model, messages=messages, tools=[], base_url=agent.base_url, api_key=agent.api_key,
        provider=agent.provider, reasoning_config={"effort": "high"}, astra_state={"base_effort": "low"} if with_effort_update else {},
    )
    request["max_output_tokens"] = 4096
    request["reasoning"]["summary"] = "concise"
    assert request["reasoning"]["effort"] == ("low" if with_effort_update else "high")
    if with_effort_update:
        assert request["input"][-2]["type"] == "configuration_update"
    started, release = threading.Event(), threading.Event()
    starts, outcomes, errors = [], [], []
    original_flush = agent._flush_messages_to_session_db

    def flush(rows, *args, **kwargs):
        if persistence_fails and any(row.get("role") == "tool" for row in rows):
            return False
        return original_flush(rows, *args, **kwargs)

    agent._flush_messages_to_session_db = flush

    def run_call(owner, dispatch, ref, **kwargs):
        rows = db.get_messages(agent.session_id)
        assert any(row.get("role") == "assistant" and row.get("tool_calls") for row in rows)
        starts.append(ref.call_id)
        started.set()
        assert release.wait(2)
        return tool_executor._ManagedToolResult("durable result", {}, [], False, True), 0.01

    monkeypatch.setattr(tool_executor, "_resolve_sequential_dispatch", lambda *args: SimpleNamespace())
    monkeypatch.setattr(tool_executor, "_run_sequential_call", run_call)
    executor = agent._astra_async_executor = AstraAsyncExecutor(agent, messages, "synthetic-task")
    socket = FakeWebSocket()
    item = {"type": "function_call", "id": "fc_live", "call_id": "call_live", "name": "read_file",
            "arguments": "{}", "async": True}

    def on_send(message, ws):
        if message["type"] == "response.create" and "previous_response_id" not in message:
            ws.push(_created("r1"))
            ws.push({"type": "response.output_item.added", "response_id": "r1", "output_index": 0, "item": item})
            ws.push({"type": "response.output_item.done", "response_id": "r1", "output_index": 0, "item": item})
        elif message["type"] == "response.steer":
            release.set()
            ws.push({"type": "response.steer.accepted", "id": "ack"})
            ws.push({"type": "response.completed", "response": {"id": "r1", "status": "completed", "output": [item]}})
            for event_id in ("pending", "duplicate-pending"):
                ws.push({"type": "response.steer.pending", "response_id": "r1", "id": event_id,
                         "required_input": [{"type": "function_call_output", "call_id": "call_live", "output": ""}]})
        elif message["type"] == "response.create":
            assert not persistence_fails
            saved = [row for row in db.get_messages(agent.session_id) if row.get("role") == "tool"]
            assert len(saved) == 1 and saved[0]["content"] == "durable result"
            assert message["input"] == [{"type": "function_call_output", "call_id": "call_live", "output": "durable result"}]
            assert message["max_output_tokens"] == 4096
            assert message["reasoning"] == request["reasoning"]
            ws.push(_created("r2"))
            for event in _completed("r2", "r2", "continued"):
                ws.push(event)

    socket._on_send = on_send
    session = AstraWebSocketSession(agent, connect=_connect_socket(socket))

    def run():
        try:
            outcomes.append(session.run(request))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert started.wait(2), errors
        assert not any(row.get("role") == "tool" for row in db.get_messages(agent.session_id))
        assert session.request_steer("continue with the result") is True
        worker.join(3)
        assert not worker.is_alive()
        assert len(starts) == 1
        continuations = [entry for entry in socket.sent if entry.get("previous_response_id") and entry["type"] == "response.create"]
        if persistence_fails:
            assert len(errors) == 1 and isinstance(errors[0], AstraProtocolError)
            assert not continuations
            assert not any(row.get("role") == "tool" for row in db.get_messages(agent.session_id))
        else:
            assert errors == [] and outcomes[0].output_text == "continued"
            assert len(continuations) == 1
            assert executor.finish_stream() is True
            assert len([row for row in db.get_messages(agent.session_id) if row.get("role") == "tool"]) == 1
    finally:
        release.set()
        session.close()
        worker.join(3)
        if not executor.closed:
            executor.abort_stream()
        agent.close()
        db.close()
