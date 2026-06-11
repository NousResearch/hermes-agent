"""Tests for per-agent local tool surfaces."""

from run_agent import AIAgent


def _local_tool():
    return {
        "name": "realtime_task_update",
        "description": "Send a realtime task progress update.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
        "handler": lambda args: {"ok": True, "summary": args["summary"]},
    }


def test_local_tools_are_visible_only_on_that_agent():
    agent = AIAgent(model="test-model", base_url="http://localhost/v1", api_key="test", enabled_toolsets=[], quiet_mode=True, local_tools=[_local_tool()])
    plain = AIAgent(model="test-model", base_url="http://localhost/v1", api_key="test", enabled_toolsets=[], quiet_mode=True)

    try:
        assert "realtime_task_update" in agent.valid_tool_names
        assert "realtime_task_update" not in plain.valid_tool_names
        assert [tool["function"]["name"] for tool in agent.tools] == ["realtime_task_update"]
        assert plain.tools == []
    finally:
        agent.close()
        plain.close()


def test_local_tool_dispatches_before_global_registry(monkeypatch):
    calls = []

    def handler(args):
        calls.append(args)
        return f"progress: {args['summary']}"

    tool = _local_tool()
    tool["handler"] = handler
    agent = AIAgent(model="test-model", base_url="http://localhost/v1", api_key="test", enabled_toolsets=[], quiet_mode=True, local_tools=[tool])

    try:
        def fail_global(*_args, **_kwargs):
            raise AssertionError("local tool leaked to global registry")

        monkeypatch.setattr("model_tools.handle_function_call", fail_global)
        result = agent._invoke_tool(
            "realtime_task_update",
            {"summary": "wired the progress hook", "status": "running"},
            "task",
            "call_1",
        )

        assert result == "progress: wired the progress hook"
        assert calls == [{"summary": "wired the progress hook", "status": "running"}]
    finally:
        agent.close()
