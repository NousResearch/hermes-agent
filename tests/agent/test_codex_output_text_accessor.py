"""Malformed SDK accessors must not crash normalization or conversation retries."""

from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import _normalize_codex_response
from tests.agent.test_run_agent_codex_responses import (
    _build_agent,
    _codex_commentary_message_response,
    _codex_message_response,
)


class RaisingOutputTextResponse:
    output = []
    status = "completed"
    incomplete_details = None
    model = "gpt-5-codex"
    usage = SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8)

    @property
    def output_text(self):
        raise TypeError("'NoneType' object is not iterable")


def test_normalization_reports_empty_output_instead_of_accessor_exception():
    with pytest.raises(RuntimeError, match="Responses API returned no output items"):
        _normalize_codex_response(RaisingOutputTextResponse())


def test_valid_output_text_fallback_remains_available():
    response = SimpleNamespace(output=[], output_text="  Recovered answer  ", status="completed")
    message, finish_reason = _normalize_codex_response(response)
    assert message.content == "Recovered answer"
    assert finish_reason == "stop"


def test_nonempty_output_with_raising_accessor_reports_no_final_answer():
    response = RaisingOutputTextResponse()
    response.output = [SimpleNamespace(type="message", content=[], status="completed")]
    message, finish_reason = _normalize_codex_response(response)
    assert message.content == ""
    assert finish_reason == "stop"


def test_auxiliary_response_preserves_structured_text_with_raising_accessor():
    from agent.auxiliary_codex_response import _parse_codex_final_response

    response = RaisingOutputTextResponse()
    response.output = _codex_message_response("Structured answer").output
    text_parts, tool_calls, usage, finish_reason = _parse_codex_final_response(response)
    assert text_parts == ["Structured answer"]
    assert tool_calls == []
    assert usage.total_tokens == 8
    assert finish_reason == "stop"


def test_commentary_output_text_does_not_become_a_final_answer():
    response = _codex_commentary_message_response("Checking the repository.")
    response.output_text = "Checking the repository."
    message, finish_reason = _normalize_codex_response(response)
    assert message.content == ""
    assert message.reasoning == "Checking the repository."
    assert finish_reason == "incomplete"


def test_run_conversation_retries_a_raising_output_text_accessor(monkeypatch):
    from agent.turn_recovery import validate_response_shape

    agent = _build_agent(monkeypatch)
    invalid, details = validate_response_shape(agent, RaisingOutputTextResponse())
    assert invalid is True
    assert details == ["response.output is empty"]
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *args, **kwargs: 0.0)
    responses = [RaisingOutputTextResponse(), _codex_message_response("Recovered answer")]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))
    result = agent.run_conversation("Give the answer")
    assert result["completed"] is True
    assert result["final_response"] == "Recovered answer"
    assert responses == []
