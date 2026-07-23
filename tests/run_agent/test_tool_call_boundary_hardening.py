from types import SimpleNamespace

import pytest

from agent.message_sanitization import (
    is_tool_call_exemplar,
    promote_text_tool_call,
    validate_tool_call_json,
)


@pytest.mark.parametrize(
    "payload",
    [
        '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>',
        '  <tool_call>\n{"name":"terminal","arguments":{"command":"pwd"}}\n</tool_call>  ',
        '{"name":"terminal","arguments":{"command":"pwd"}}',
    ],
)
def test_validate_tool_call_json_accepts_one_complete_object(payload: str) -> None:
    valid, parsed, error = validate_tool_call_json(payload)

    assert valid is True
    assert parsed == {"name": "terminal", "arguments": {"command": "pwd"}}
    assert error is None
    assert is_tool_call_exemplar(payload) is True


@pytest.mark.parametrize(
    "payload",
    [
        'prefix <tool_call>{"name":"terminal","arguments":{}}</tool_call>',
        '<tool_call>{"name":"terminal","arguments":{}}</tool_call> suffix',
        '<tool_call>{"name":"terminal","arguments":{}}</tool_call>'
        '<tool_call>{"name":"terminal","arguments":{}}</tool_call>',
        '<tool_call><tool_call>{"name":"terminal","arguments":{}}</tool_call></tool_call>',
        '<tool_call>{"name":"terminal","arguments":"{}"}</tool_call>',
        '<tool_call>{"name":" terminal ","arguments":{}}</tool_call>',
    ],
)
def test_validate_tool_call_json_rejects_ambiguous_or_unsafe_wrappers(
    payload: str,
) -> None:
    valid, parsed, error = validate_tool_call_json(payload)

    assert valid is False
    assert parsed is None
    assert error
    assert is_tool_call_exemplar(payload) is False


def _repair(tool_names: set[str], emitted_name: str) -> str | None:
    from run_agent import AIAgent

    stub = SimpleNamespace(valid_tool_names=tool_names)
    repair = AIAgent._repair_tool_call.__get__(stub, AIAgent)
    return repair(emitted_name)


def test_exact_shell_alias_resolves_to_registered_terminal() -> None:
    assert _repair({"terminal", "read_file"}, "shell") == "terminal"


def test_shell_alias_is_case_sensitive_and_fails_closed_without_terminal() -> None:
    assert _repair({"terminal", "read_file"}, "Shell") is None
    assert _repair({"bash", "read_file"}, "shell") is None


def test_promote_text_tool_call_builds_structured_call() -> None:
    message = SimpleNamespace(
        content='<tool_call>{"name":"shell","arguments":{"command":"pwd"}}</tool_call>',
        tool_calls=None,
    )

    assert promote_text_tool_call(message) is True
    assert message.content == ""
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].type == "function"
    assert message.tool_calls[0].function.name == "shell"
    assert message.tool_calls[0].function.arguments == '{"command":"pwd"}'


def test_promote_text_tool_call_does_not_override_or_parse_prose() -> None:
    existing = [SimpleNamespace(id="existing")]
    message = SimpleNamespace(
        content='prefix <tool_call>{"name":"shell","arguments":{}}</tool_call>',
        tool_calls=existing,
    )

    assert promote_text_tool_call(message) is False
    assert message.tool_calls is existing
