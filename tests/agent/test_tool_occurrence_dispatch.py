"""Both native executor modes bind the durable occurrence before registry dispatch."""

from types import SimpleNamespace

import pytest

from agent.tool_call_context import CURRENT_TOOL_CALL_UID
from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential
from hermes_state import SessionDB
from tests.agent.test_sequential_tool_timeout import _make_agent, _tool_call


@pytest.mark.parametrize("execute", [execute_tool_calls_sequential, execute_tool_calls_concurrent])
def test_native_executor_binds_persisted_occurrences(execute, tmp_path, monkeypatch):
    import model_tools

    agent = _make_agent(tmp_path)
    calls = [_tool_call("first"), _tool_call("second")]
    assistant = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call.id, "type": "function", "function": {"name": "web_extract", "arguments": "{}"}} for call in calls]}
    db = SessionDB(db_path=tmp_path / "occurrences.db")
    try:
        db.create_session("proof", "api_server")
        db.append_messages_batch("proof", [assistant])
    finally:
        db.close()
    received = []

    def dispatch(name, args, **kwargs):
        received.append(CURRENT_TOOL_CALL_UID.get())
        return '{"result":"ok"}'

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    execute(agent, SimpleNamespace(tool_calls=calls), [assistant], "task")
    assert sorted(received) == sorted(assistant["_tool_call_uids"].values())
    assert CURRENT_TOOL_CALL_UID.get() == ""
