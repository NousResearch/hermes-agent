"""Dropped streams must not dispatch repetition-dominated tool arguments."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID
from run_agent import AIAgent


REPEATED_COMMAND = "cat >> /tmp/example.py <<'PYEOF'\n" + (
    "# The model repeats this same rambling fragment instead of writing the script.\n" * 150
) + "PYEOF"


def _stream_response(monkeypatch, arguments, finish_reason=None):
    agent = AIAgent(
        api_key="test-key", base_url="https://example.com/v1", model="test/model",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    def chunk(calls=None, reason=None):
        delta = SimpleNamespace(content=None, tool_calls=calls, reasoning_content=None, reasoning=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(index=0, delta=delta, finish_reason=reason)],
            model=None, usage=None,
        )

    def stream():
        # A complete sibling must not run if the same action batch is discarded.
        yield chunk([SimpleNamespace(index=0, id="sibling", function=SimpleNamespace(
            name="read_file", arguments='{"path":"/tmp/example.py"}'))])
        yield chunk([SimpleNamespace(index=1, id="write", function=SimpleNamespace(
            name="terminal", arguments=arguments))])
        if finish_reason:
            yield chunk(reason=finish_reason)

    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kwargs: stream()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kwargs: None)
    return agent._interruptible_streaming_api_call({})


@pytest.mark.parametrize("repair_needed", [False, True], ids=["valid-json", "repairable-json"])
def test_dropped_repetitive_arguments_never_reach_dispatch(monkeypatch, repair_needed):
    arguments = json.dumps({"command": REPEATED_COMMAND})
    if repair_needed:
        arguments = arguments[:-1]  # Missing closing object, repairable by the existing helper.
    response = _stream_response(monkeypatch, arguments)
    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.tool_calls is None
    assert set(response._dropped_tool_names) == {"read_file", "terminal"}


@pytest.mark.parametrize("command,finish_reason", [
    ("pwd", None),
    ("\n".join(f"echo unique row {i}: value {i * 7}" for i in range(150)), None),
    (REPEATED_COMMAND, "tool_calls"),
])
def test_healthy_or_confirmed_arguments_are_preserved(monkeypatch, command, finish_reason):
    arguments = json.dumps({"command": command})
    response = _stream_response(monkeypatch, arguments, finish_reason)
    assert response.id != PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.tool_calls[1].function.arguments == arguments
