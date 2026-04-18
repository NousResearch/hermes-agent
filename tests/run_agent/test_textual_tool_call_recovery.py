import json
from types import SimpleNamespace


def _chat_response(content: str, finish_reason: str = "stop"):
    assistant = SimpleNamespace(content=content, reasoning=None, tool_calls=[])
    choice = SimpleNamespace(message=assistant, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


class _SequentialCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        index = len(self.requests) - 1
        return self._responses[index]


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_SequentialCompletions(responses))


def _make_agent(monkeypatch, responses):
    from run_agent import AIAgent

    fake_client = _FakeClient(responses)
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )

    invocations = []

    def _fake_handle_function_call(name, args, task_id=None, **kwargs):
        invocations.append((name, args))
        return json.dumps({"ok": True, "name": name, "args": args})

    monkeypatch.setattr("run_agent.handle_function_call", _fake_handle_function_call)

    agent = AIAgent(
        model="minimax-m2.7",
        api_key="test-key",
        base_url="http://localhost:4000/v1",
        platform="telegram",
        max_iterations=4,
        quiet_mode=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    return agent, fake_client.chat.completions, invocations


def test_recovers_minimax_namespaced_tool_call_and_executes_tool(monkeypatch):
    first_turn = (
        "<minimax:tool_call>"
        '{"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
        "</minimax:tool_call>"
    )
    responses = [
        _chat_response(first_turn),
        _chat_response("done"),
    ]
    agent, completions, invocations = _make_agent(monkeypatch, responses)

    result = agent.run_conversation("read the file")

    assert result["final_response"] == "done"
    assert invocations == [("read_file", {"path": "README.md"})]

    second_messages = completions.requests[1]["messages"]
    assistant_contents = [
        msg.get("content", "")
        for msg in second_messages
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    ]
    assert all("<minimax:tool_call" not in (text or "").lower() for text in assistant_contents)


def test_recovers_invoke_tag_with_name_attribute(monkeypatch):
    responses = [
        _chat_response('<invoke name="read_file">{"path":"AGENTS.md"}</invoke>'),
        _chat_response("ok"),
    ]
    agent, _, invocations = _make_agent(monkeypatch, responses)

    result = agent.run_conversation("read agents file")

    assert result["final_response"] == "ok"
    assert invocations == [("read_file", {"path": "AGENTS.md"})]


def test_recovers_unclosed_invoke_tag_at_end_of_message(monkeypatch):
    responses = [
        _chat_response('<invoke name="read_file">{"path":"README.md"}'),
        _chat_response("resolved"),
    ]
    agent, _, invocations = _make_agent(monkeypatch, responses)

    result = agent.run_conversation("read readme")

    assert result["final_response"] == "resolved"
    assert invocations == [("read_file", {"path": "README.md"})]
