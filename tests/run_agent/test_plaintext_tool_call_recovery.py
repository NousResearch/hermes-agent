"""Tests for recovering tool-call-shaped JSON emitted as assistant text."""

import json
from types import SimpleNamespace

from agent.message_sanitization import recover_plaintext_tool_call


CLARIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "clarify",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
        },
    },
}

TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}


def test_recovers_explicit_action_envelope_as_tool_call():
    recovered = recover_plaintext_tool_call(
        '{"action":"clarify","question":"Pick one","choices":["A","B"]}',
        [CLARIFY_TOOL],
    )

    assert recovered is not None
    assert recovered["name"] == "clarify"
    assert json.loads(recovered["arguments"]) == {
        "question": "Pick one",
        "choices": ["A", "B"],
    }


def test_recovers_fenced_action_envelope():
    recovered = recover_plaintext_tool_call(
        '```json\n{"action":"clarify","question":"Pick one"}\n```',
        [CLARIFY_TOOL],
    )

    assert recovered is not None
    assert recovered["name"] == "clarify"
    assert json.loads(recovered["arguments"]) == {"question": "Pick one"}


def test_does_not_recover_unmarked_json_answer():
    assert recover_plaintext_tool_call(
        '{"question":"What did the report ask?","answer":"Pick A"}',
        [CLARIFY_TOOL],
    ) is None


def test_does_not_recover_when_required_arguments_missing():
    assert recover_plaintext_tool_call(
        '{"action":"clarify","choices":["A","B"]}',
        [CLARIFY_TOOL],
    ) is None


def test_does_not_recover_non_allowlisted_tool_envelope():
    assert recover_plaintext_tool_call(
        '{"action":"terminal","command":"echo should-not-run"}',
        [TERMINAL_TOOL],
    ) is None


def _response(content: str, *, finish_reason: str = "stop"):
    assistant = SimpleNamespace(content=content, reasoning=None, tool_calls=[])
    choice = SimpleNamespace(message=assistant, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None, model="test/model")


class _FakeChatCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _response(
                '{"action":"clarify","question":"Pick one","choices":["A","B"]}'
            )
        return _response("Thanks, continuing with A.")


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_loop_dispatches_recovered_plaintext_tool_call(monkeypatch):
    from run_agent import AIAgent

    clarify_calls = []

    def _clarify(question, choices):
        clarify_calls.append((question, choices))
        return "A"

    fake_client = _FakeClient()
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [CLARIFY_TOOL],
    )

    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform="telegram",
        max_iterations=3,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
        clarify_callback=_clarify,
    )
    agent._disable_streaming = True

    result = agent.run_conversation("ambiguous request")

    assert clarify_calls == [("Pick one", ["A", "B"])]
    assert result["final_response"].startswith("Thanks")
    assert '{"action":"clarify"' not in result["final_response"]
