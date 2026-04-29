import json
from types import SimpleNamespace


def _tool_call(name: str, arguments):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response_with_tool_call(arguments):
    assistant = SimpleNamespace(
        content=None,
        reasoning=None,
        tool_calls=[_tool_call("read_file", arguments)],
    )
    choice = SimpleNamespace(message=assistant, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


class _FakeChatCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _response_with_tool_call({"path": "README.md"})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", reasoning=None, tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_tool_call_validation_accepts_dict_arguments(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: _FakeClient())
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True, "args": args}),
    )

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=3,
        quiet_mode=True,
        skip_memory=True,
    )
    agent._disable_streaming = True

    result = agent.run_conversation("read the file")

    assert result["final_response"] == "done"
