import asyncio

from agent.claude_sdk_mcp import build_hermes_sdk_mcp_server


class FakeSdk:
    @staticmethod
    def tool(name, description, input_schema):
        def decorate(handler):
            handler.sdk_name = name
            handler.sdk_description = description
            handler.sdk_input_schema = input_schema
            return handler

        return decorate

    @staticmethod
    def create_sdk_mcp_server(*, name, version, tools):
        return {"name": name, "version": version, "tools": tools}


def _definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
            },
        },
    }


def test_in_process_server_uses_authoritative_schema_and_worker_task_id():
    calls = []

    def dispatch(name, arguments, *, task_id):
        calls.append((name, arguments, task_id))
        return '{"success": true}'

    server = build_hermes_sdk_mcp_server(
        [_definition("kanban_complete"), _definition("web_search")],
        dispatch=dispatch,
        task_id="BUILD-392",
        sdk=FakeSdk,
        allowed_names={"kanban_complete"},
    )

    assert server["name"] == "hermes"
    assert [tool.sdk_name for tool in server["tools"]] == ["kanban_complete"]
    assert server["tools"][0].sdk_input_schema == _definition("kanban_complete")[
        "function"
    ]["parameters"]

    result = asyncio.run(server["tools"][0]({"summary": "done"}))

    assert calls == [("kanban_complete", {"summary": "done"}, "BUILD-392")]
    assert result == {"content": [{"type": "text", "text": '{"success": true}'}]}


def test_in_process_server_returns_structured_tool_errors():
    def dispatch(name, arguments, *, task_id):
        raise RuntimeError("database unavailable")

    server = build_hermes_sdk_mcp_server(
        [_definition("kanban_complete")],
        dispatch=dispatch,
        task_id="BUILD-392",
        sdk=FakeSdk,
    )

    result = asyncio.run(server["tools"][0]({"summary": "done"}))

    assert result["is_error"] is True
    assert "database unavailable" in result["content"][0]["text"]


def test_terminal_arguments_are_transformed_before_authoritative_dispatch():
    calls = []

    def dispatch(name, arguments, *, task_id):
        calls.append((name, arguments, task_id))
        return "ok"

    server = build_hermes_sdk_mcp_server(
        [_definition("terminal")],
        dispatch=dispatch,
        task_id="worker-run",
        sdk=FakeSdk,
        allowed_names={"terminal"},
        argument_transform=lambda name, args: {
            **args,
            "command": f"sandboxed:{args['command']}",
        },
    )

    result = asyncio.run(server["tools"][0]({"command": "pytest -q"}))

    assert result["content"][0]["text"] == "ok"
    assert calls == [
        ("terminal", {"command": "sandboxed:pytest -q"}, "worker-run")
    ]
