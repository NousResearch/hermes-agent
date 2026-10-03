"""Plugin replacements survive while changes to successful search results are observable."""

from copy import deepcopy
import json
import logging
from types import SimpleNamespace

import pytest

import model_tools
from agent.tool_dispatch_helpers import make_tool_result_message
from hermes_cli import plugins
from run_agent import AIAgent


_CANARY = "search-content-canary"
_SUCCESS = json.dumps({"success": True, "data": {"items": [_CANARY], "flag": True, "count": 1}})
_REDACTED = json.dumps({"success": True, "data": {"items": ["[REDACTED]"], "flag": True, "count": 1}})
_FAILURE = json.dumps({"success": False, "error": _CANARY})
_CASES = [
    pytest.param("web_search", _SUCCESS, _SUCCESS, "replace", False, id="unchanged"),
    pytest.param("web_search", _SUCCESS,
                 '{"data":{"count":1,"flag":true,"items":["search-content-canary"]},"success":true}',
                 "replace", False, id="format-and-key-order"),
    pytest.param("web_search", _SUCCESS, _REDACTED, "replace", True, id="mandatory-redaction"),
    pytest.param("web_search", _FAILURE, _REDACTED, "replace", True, id="failure-claims-success"),
    pytest.param("web_search", _SUCCESS, _SUCCESS.replace('"flag": true', '"flag": 1'),
                 "replace", True, id="boolean-is-not-integer"),
    pytest.param("web_search", _SUCCESS, _SUCCESS.replace('"count": 1', '"count": 1.0'),
                 "replace", True, id="integer-is-not-float"),
    pytest.param("web_search", _SUCCESS, _CANARY, "replace", True, id="non-json-replacement"),
    pytest.param("web_search", _SUCCESS, '{"success":true,"data":NaN}',
                 "replace", True, id="nonstandard-json-replacement"),
    pytest.param("web_search", _SUCCESS, '{"success":true,"success":true}',
                 "replace", True, id="duplicate-json-replacement"),
    pytest.param("web_search", '{"success":true,"data":NaN}', _REDACTED,
                 "replace", True, id="nonstandard-json-claims-success"),
    pytest.param("web_search", '{"success":false,"success":true}', _REDACTED,
                 "replace", True, id="duplicate-json-claims-success"),
    pytest.param("web_search", _CANARY, _REDACTED, "replace", True, id="non-json-claims-success"),
    pytest.param("web_search", _FAILURE, '{"success":false,"error":"[REDACTED]"}',
                 "replace", False, id="failure-redaction"),
    pytest.param("web_search", _SUCCESS, "", "replace", True, id="empty-replacement"),
    pytest.param("web_extract", _SUCCESS, _REDACTED, "replace", False, id="other-tool"),
]
_EXECUTION_CASES = [
    *_CASES,
    pytest.param("web_search", _SUCCESS, _REDACTED, "short-circuit", True, id="success-without-dispatch"),
    pytest.param("web_search", _SUCCESS, _FAILURE, "short-circuit", False, id="failure-without-dispatch"),
    pytest.param("web_search", json.loads(_SUCCESS), json.loads(_REDACTED),
                 "in-place", True, id="mutable-dispatch-result"),
]


@pytest.fixture
def plugin_context(monkeypatch):
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return plugins.PluginContext(plugins.PluginManifest(name="result-redaction", source="user"), manager)


@pytest.fixture
def search_agent(monkeypatch):
    definitions = [{"type": "function", "function": {
        "name": name, "description": "search fixture", "parameters": {"type": "object", "properties": {}},
    }} for name in ("web_search", "web_extract")]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kwargs: definitions)
    monkeypatch.setattr(model_tools, "check_toolset_requirements", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *_args, **_kwargs: 200000)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 200000)
    agent = AIAgent(
        model="test-model", provider="custom", api_key="test-key", base_url="http://model.example.test/v1",
        enabled_toolsets=["web"], quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False,
    )
    try:
        yield agent
    finally:
        agent.close()


def _execute_search(entrypoint, agent, tool_name):
    args = {"query": "query-canary", "url": "https://content.example.test/private"}
    if entrypoint == "direct":
        return model_tools.handle_function_call(tool_name, args, skip_pre_tool_call_hook=True,
                                                session_id="session-canary", tool_call_id="search-call")
    call = SimpleNamespace(id="search-call", type="function",
                           function=SimpleNamespace(name=tool_name, arguments=json.dumps(args)))
    assistant = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    execute = (agent._execute_tool_calls_concurrent if entrypoint == "concurrent"
               else agent._execute_tool_calls_sequential)
    execute(assistant, messages, "search-task")
    assert len(messages) == 1 and messages[0]["role"] == "tool"
    return messages[0]["content"]


def _assert_audit(caplog, source, expected):
    warnings = [record.getMessage() for record in caplog.records
                if record.levelno >= logging.WARNING and source in record.getMessage()
                and "web_search" in record.getMessage()]
    assert bool(warnings) is expected
    assert len(warnings) <= 1
    for message in warnings:
        for private_value in (_CANARY, "query-canary", "content.example.test", "session-canary", "search-call"):
            assert private_value not in message


@pytest.mark.parametrize("entrypoint", ["direct", "sequential", "concurrent"])
@pytest.mark.parametrize("tool_name,original,candidate,mode,expected", _EXECUTION_CASES)
def test_execution_middleware_audits_without_replacing_plugin_output(
    monkeypatch, caplog, plugin_context, search_agent, entrypoint, tool_name, original, candidate, mode, expected,
):
    original, candidate = deepcopy(original), deepcopy(candidate)
    dispatched = []

    def dispatch(name, args, **_kwargs):
        dispatched.append((name, args))
        return original

    def middleware(next_call, **_kwargs):
        if mode == "short-circuit":
            return candidate
        downstream = next_call()
        if mode == "in-place":
            downstream.clear()
            downstream.update(candidate)
            return json.dumps(downstream)
        return candidate

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    plugin_context.register_middleware("tool_execution", middleware)
    with caplog.at_level(logging.WARNING):
        output = _execute_search(entrypoint, search_agent, tool_name)
    expected_output = json.dumps(candidate) if isinstance(candidate, dict) else candidate
    if entrypoint != "direct":
        expected_output = make_tool_result_message(tool_name, expected_output, "search-call")["content"]
    assert output == expected_output
    assert len(dispatched) == (0 if mode == "short-circuit" else 1)
    _assert_audit(caplog, "tool_execution middleware", expected)


@pytest.mark.parametrize("tool_name,original,candidate,mode,expected", _CASES)
def test_transform_hook_audits_without_disabling_redaction(
    monkeypatch, caplog, plugin_context, tool_name, original, candidate, mode, expected,
):
    dispatched, observed = [], []

    def dispatch(name, args, **_kwargs):
        dispatched.append((name, args))
        return original

    def transform(result, **_kwargs):
        observed.append(result)
        return candidate

    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())
    plugin_context.register_hook("transform_tool_result", lambda **_kwargs: None)
    plugin_context.register_hook("transform_tool_result", transform)
    plugin_context.register_hook("transform_tool_result", lambda **_kwargs: "later result must not win")
    with caplog.at_level(logging.WARNING):
        output = _execute_search("direct", None, tool_name)
    assert output == candidate
    assert observed == [original]
    assert len(dispatched) == 1
    _assert_audit(caplog, "transform_tool_result hook", expected)
