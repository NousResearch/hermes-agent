"""On the agent path, ``pre_tool_call`` hooks see the same coerced arguments the tool runs with.

A model can send an array argument as a JSON string. The registry coerces it before the tool runs,
so a policy hook that judged the raw string let through a call the tool then executed as a list.
"""
import json
from types import SimpleNamespace


def test_pre_tool_call_hook_sees_the_coerced_array_and_its_block_holds(monkeypatch):
    from agent.tool_executor import _parse_tool_call, _pre_tool_block
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    seen = []

    def lifecycle(hook_name, **kw):  # a policy plugin's pre_tool_call callback, as the dispatcher runs it
        todos = kw["args"].get("todos")
        seen.append(type(todos).__name__)
        if isinstance(todos, list) and len(todos) > 1:
            return [{"action": "block", "message": "policy: one todo at a time"}]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lifecycle)
    todos = [{"id": "1", "content": "a", "status": "pending"}, {"id": "2", "content": "b", "status": "pending"}]
    call = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(
        name="todo", arguments=json.dumps({"todos": json.dumps(todos)})))  # array sent as a JSON string
    agent = SimpleNamespace(session_id="s", _current_turn_id="t", _api_request_id="r")

    parsed = _parse_tool_call(agent, call)
    block, final_args = _pre_tool_block(agent, parsed.ref("task"))

    assert seen == ["list"]
    assert block == "policy: one todo at a time"
    assert final_args["todos"] == todos


def test_nested_json_strings_are_fully_repaired_before_the_hook_runs():
    """``questions`` and each question's ``choices`` both sent as JSON strings: the hook must see the
    list of choices, and the registry's later pass must leave the arguments unchanged."""
    from agent.tool_executor import _parse_tool_call
    from tools.arg_coercion import coerce_tool_args
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    choices = ["red", "green", "blue"]
    questions = [{"question": "Pick a colour", "choices": json.dumps(choices)}]
    call = SimpleNamespace(id="c1", type="function", function=SimpleNamespace(
        name="clarify", arguments=json.dumps({"questions": json.dumps(questions)})))
    agent = SimpleNamespace(session_id="s", _current_turn_id="t", _api_request_id="r")

    parsed_args = _parse_tool_call(agent, call).args

    assert parsed_args["questions"] == [{"question": "Pick a colour", "choices": choices}]
    assert coerce_tool_args("clarify", json.loads(json.dumps(parsed_args))) == parsed_args
