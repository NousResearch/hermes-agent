import json
from types import SimpleNamespace

from agent.tool_aliases import normalize_tool_call_aliases
from agent.transports.types import ToolCall


def _args(call):
    return json.loads(call.function.arguments)


def test_normalizes_write_alias_to_write_file():
    call = ToolCall(
        id="call_1",
        name="write",
        arguments=json.dumps({"path": "/tmp/example.txt", "fileText": "hello"}),
    )

    normalize_tool_call_aliases([call], {"write_file"})

    assert call.function.name == "write_file"
    assert _args(call) == {"path": "/tmp/example.txt", "content": "hello"}


def test_normalizes_shell_alias_to_terminal():
    call = ToolCall(
        id="call_1",
        name="shell",
        arguments=json.dumps({"command": "pwd", "workingDirectory": "/repo", "timeout": 30}),
    )

    normalize_tool_call_aliases([call], {"terminal"})

    assert call.function.name == "terminal"
    assert _args(call) == {"command": "pwd", "workdir": "/repo", "timeout": 30}


def test_leaves_alias_when_target_tool_unavailable():
    call = ToolCall(
        id="call_1",
        name="write",
        arguments=json.dumps({"path": "/tmp/example.txt", "fileText": "hello"}),
    )

    normalize_tool_call_aliases([call], {"terminal"})

    assert call.function.name == "write"


def test_supports_openai_like_tool_call_objects():
    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="grep",
            arguments=json.dumps({"pattern": "foo", "glob": "*.py", "contextBefore": 2}),
        ),
    )

    normalize_tool_call_aliases([call], {"search_files"})

    assert call.function.name == "search_files"
    assert json.loads(call.function.arguments) == {
        "pattern": "foo",
        "file_glob": "*.py",
        "context": 2,
    }
