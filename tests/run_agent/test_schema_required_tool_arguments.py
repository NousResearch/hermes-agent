"""Regression coverage for schema-required tool arguments at dispatch."""

from types import SimpleNamespace

import pytest


def _tool_call(arguments: str):
    return SimpleNamespace(
        id="call-required",
        type="function",
        function=SimpleNamespace(name="read_file", arguments=arguments),
    )


def _response_with_tool_call(arguments: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning=None,
                    tool_calls=[_tool_call(arguments)],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


class _FakeChatCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= 3:
            return _response_with_tool_call("{}")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="done", reasoning=None, tool_calls=[]
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def _validation_agent(parameters: dict):
    return SimpleNamespace(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": parameters,
                },
            }
        ],
        valid_tool_names={"read_file"},
    )


def _validation_call(arguments):
    return SimpleNamespace(
        function=SimpleNamespace(name="read_file", arguments=arguments)
    )


@pytest.mark.parametrize("arguments", ["", "{}"])
def test_required_schema_rejects_empty_arguments(arguments):
    from agent.conversation_loop import _validate_tool_call_arguments

    invalid = _validate_tool_call_arguments(
        _validation_agent({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }),
        [_validation_call(arguments)],
    )

    assert len(invalid) == 1
    assert "path" in invalid[0][1]


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "object", "properties": {}, "required": []},
        {
            "type": "object",
            "properties": {"mode": {"type": "string"}},
        },
    ],
)
def test_empty_object_is_allowed_without_top_level_required_fields(parameters):
    from agent.conversation_loop import _validate_tool_call_arguments

    call = _validation_call("{}")
    assert _validate_tool_call_arguments(_validation_agent(parameters), [call]) == []
    assert call.function.arguments == "{}"


def test_valid_required_arguments_are_allowed():
    from agent.conversation_loop import _validate_tool_call_arguments

    assert (
        _validate_tool_call_arguments(
            _validation_agent({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
            [_validation_call('{"path":"README.md"}')],
        )
        == []
    )


def test_missing_one_of_multiple_required_fields_is_rejected():
    from agent.conversation_loop import _validate_tool_call_arguments

    invalid = _validate_tool_call_arguments(
        _validation_agent({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "mode": {"type": "string"},
            },
            "required": ["path", "mode"],
        }),
        [_validation_call('{"path":"README.md"}')],
    )

    assert len(invalid) == 1
    assert "mode" in invalid[0][1]
    assert "path" not in invalid[0][1]


def test_nested_required_is_outside_this_top_level_boundary():
    from agent.conversation_loop import _validate_tool_call_arguments

    parameters = {
        "type": "object",
        "properties": {
            "options": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
        },
        "required": [],
    }

    assert (
        _validate_tool_call_arguments(
            _validation_agent(parameters), [_validation_call("{}")]
        )
        == []
    )


def test_malformed_non_empty_arguments_remain_on_existing_recovery_path():
    from agent.conversation_loop import _validate_tool_call_arguments

    call = _validation_call('{"path":')
    assert (
        _validate_tool_call_arguments(
            _validation_agent({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
            [call],
        )
        == []
    )
    assert call.function.arguments == '{"path":'


def test_required_schema_empty_object_never_reaches_handler(monkeypatch):
    from run_agent import AIAgent

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    executed = []

    def fake_dispatch(name, args, task_id=None, **kwargs):
        executed.append((name, args))
        return "unexpected handler execution"

    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: _FakeClient())
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda *a, **k: tool_defs)
    monkeypatch.setattr("run_agent.handle_function_call", fake_dispatch)

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=6,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
    )
    setattr(agent, "_disable_streaming", True)

    result = agent.run_conversation("invoke the file tool")

    assert executed == []
    assert result["final_response"].startswith("done")
    tool_results = [
        message for message in result["messages"] if message.get("role") == "tool"
    ]
    assert len(tool_results) == 1
    assert "missing required fields" in tool_results[0]["content"]
