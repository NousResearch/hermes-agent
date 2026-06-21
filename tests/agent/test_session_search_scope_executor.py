import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import invoke_tool
from gateway.session_context import clear_session_vars, set_session_vars


class _Agent(SimpleNamespace):
    def _get_session_db_for_recall(self):
        return object()


def test_concurrent_invoke_tool_passes_hidden_scope_and_profile(monkeypatch):
    captured = {}

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True})

    monkeypatch.setattr("tools.session_search_tool.session_search", fake_session_search)

    agent = _Agent(
        session_id="session-a",
        _gateway_session_key=None,
        _gateway_recall_scope_key=None,
        _current_turn_id="",
        _current_api_request_id="",
    )
    tokens = set_session_vars(
        platform="discord",
        session_key="agent:main:discord:channel:now:user-a",
        recall_scope_key="agent:main:discord:channel:now",
    )
    try:
        result = invoke_tool(
            agent,
            "session_search",
            {"query": "周文档", "profile": "work"},
            effective_task_id="task-a",
            pre_tool_block_checked=True,
            skip_tool_request_middleware=True,
        )
    finally:
        clear_session_vars(tokens)

    assert json.loads(result) == {"success": True}
    assert captured["query"] == "周文档"
    assert captured["profile"] == "work"
    assert captured["current_scope_key"] == "agent:main:discord:channel:now"
    assert captured["gateway_context"] is True


def test_concurrent_invoke_tool_stays_global_without_gateway_context(monkeypatch):
    captured = {}

    def fake_session_search(**kwargs):
        captured.update(kwargs)
        return json.dumps({"success": True})

    monkeypatch.setattr("tools.session_search_tool.session_search", fake_session_search)

    agent = _Agent(
        session_id="session-a",
        _gateway_session_key="agent:main:discord:channel:now:user-a",
        _gateway_recall_scope_key="agent:main:discord:channel:now",
        _current_turn_id="",
        _current_api_request_id="",
    )

    result = invoke_tool(
        agent,
        "session_search",
        {"query": "周文档"},
        effective_task_id="task-a",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
    )

    assert json.loads(result) == {"success": True}
    assert captured["current_scope_key"] is None
    assert captured["gateway_context"] is False
