"""Hook parity across real sequential/concurrent registry and connector dispatch."""

import json

import pytest
import model_tools
from tests.agent.test_run_agent import (  # noqa: F401
    agent, _mock_assistant_msg, _mock_tool_call, _mock_plugin_discovery,
)


def _record_hooks(monkeypatch):
    events = []

    def invoke(hook, **kwargs):
        if hook in {"post_tool_call", "transform_tool_result"}:
            events.append((hook, kwargs["tool_name"], kwargs["result"]))
        if hook == "transform_tool_result":
            return [json.dumps({"response": f"T[{kwargs['tool_name']}]({kwargs['result']})"})]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _: True)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke)
    return events


def _run(agent, executor, name, args):
    messages = []
    call = _mock_tool_call(name=name, arguments=json.dumps(args), call_id="parity-call")
    getattr(agent, f"_execute_tool_calls_{executor}")(
        _mock_assistant_msg(content="", tool_calls=[call]), messages, "task-1"
    )
    return messages[0]["content"]


@pytest.mark.parametrize("executor", ["sequential", "concurrent"])
def test_connector_batch_hooks_are_per_entry(agent, monkeypatch, executor):
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    agent.enabled_toolsets = ["connections"]
    names = ["connectors__gmail__FETCH_EMAILS", "connectors__slack__FETCH_MESSAGES"]
    calls = [{"name": name, "arguments": {}} for name in names]
    wire = []

    class Client:
        def execute(self, planned):
            wire.extend(p.name for p in planned)
            return [{"data": f"raw:{p.name}", "error": None} for p in planned]

    monkeypatch.setattr(bridge, "_default_client_factory", Client)
    events = _record_hooks(monkeypatch)
    content = _run(agent, executor, "tool_call", {"calls": calls})
    expected = []
    for name in names:
        raw = json.dumps({"response": f"raw:{name}"}, ensure_ascii=False)
        expected.extend([(hook, name, raw) for hook in ("post_tool_call", "transform_tool_result")])
    expected.append(("post_tool_call", "tool_call", content))
    assert events == expected
    assert wire == names
    result = json.loads(content)
    assert [entry["response"] for entry in result["results"]] == [
        f'T[{name}]({json.dumps({"response": f"raw:{name}"})})' for name in names
    ]


@pytest.mark.parametrize("executor", ["sequential", "concurrent"])
@pytest.mark.parametrize("outcome", ["approval_block", "dispatch_exception"])
def test_guard_and_dispatch_exception_skip_transform(agent, monkeypatch, executor, outcome):
    events = _record_hooks(monkeypatch)
    executed = []

    def execute(*args, **kwargs):
        executed.append(args[0])
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(model_tools, "_execute_tool", execute)
    if outcome == "approval_block":
        monkeypatch.setattr("acp_adapter.edit_approval.maybe_require_edit_approval",
                            lambda *args: '{"error": "approval denied"}')
    content = _run(agent, executor, "web_search", {})
    raw = ('{"error": "approval denied"}' if outcome == "approval_block" else
           json.dumps({"error": "[TOOL_ERROR] Error executing web_search: dispatch failed"}))
    assert events == [("post_tool_call", "web_search", raw)]
    assert raw in content
    assert executed == ([] if outcome == "approval_block" else ["web_search"])
