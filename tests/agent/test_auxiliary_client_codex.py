from types import SimpleNamespace

from agent.auxiliary_client import _CodexCompletionsAdapter


class _FakeStream:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(())

    def get_final_response(self):
        return SimpleNamespace(output=[])


def test_codex_adapter_flattens_tool_role_messages_for_responses_input():
    stream = _FakeStream()
    real_client = SimpleNamespace(responses=SimpleNamespace(stream=stream))
    adapter = _CodexCompletionsAdapter(real_client, "gpt-5.2-codex")

    adapter.create(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "run a tool"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_123", "content": "tool output"},
        ],
    )

    assert stream.kwargs["instructions"] == "sys"
    assert [m["role"] for m in stream.kwargs["input"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert stream.kwargs["input"][-1]["content"] == "[tool result call_123: tool output]"


def test_codex_adapter_flattens_function_role_messages_for_responses_input():
    stream = _FakeStream()
    real_client = SimpleNamespace(responses=SimpleNamespace(stream=stream))
    adapter = _CodexCompletionsAdapter(real_client, "gpt-5.2-codex")

    adapter.create(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "call legacy function"},
            {"role": "function", "name": "legacy_fn", "content": "legacy output"},
        ],
    )

    assert stream.kwargs["instructions"] == "sys"
    assert [m["role"] for m in stream.kwargs["input"]] == ["user", "user"]
    assert stream.kwargs["input"][-1]["content"] == "[function result legacy_fn: legacy output]"
