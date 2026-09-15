"""Invariant tests for in-stream retry of a terminal ``response.failed`` on the Responses path.

A provider can end a stream with ``response.failed`` and a transient error code
(``server_error``). That frame produces no answer, so the stream is safe to retry once rather than
surfacing a hard failure the outer turn loop has to absorb. The contracts pinned here:

- a retryable terminal failure retries in-stream and returns the later attempt's response;
- the retry is bounded, so a provider that keeps failing still surfaces the failure;
- ``status=incomplete`` never retries (a retry would re-burn the completion budget for the same
  truncation);
- a non-retryable failed code never retries (a permanent failure would fail identically);
- a superseded attempt's streamed text does not leak into the retry's accounting.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent._interrupt_requested = False
    return agent


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def close(self):
        return None


def _failed_frame(code="server_error", message="The model failed to generate a response.", text=None):
    events = []
    if text is not None:
        events.append(SimpleNamespace(type="response.output_text.delta", delta=text))
    events.append(SimpleNamespace(
        type="response.failed",
        response=SimpleNamespace(
            status="failed", id="resp_failed", usage=None,
            error=SimpleNamespace(code=code, message=message),
        ),
    ))
    return events


def _completed_frame(text="answer"):
    return [
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed", id="resp_ok", usage=None,
                output=[SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text=text)],
                )],
            ),
        ),
    ]


def _incomplete_frame():
    return [
        SimpleNamespace(
            type="response.incomplete",
            response=SimpleNamespace(
                status="incomplete", id="resp_inc", usage=None,
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            ),
        ),
    ]


def _run(agent, streams):
    """Drive _run_codex_stream with one fake SSE stream per attempt; return (result, call_count)."""
    calls = {"n": 0}

    def _create(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        return _FakeStream(streams[i] if i < len(streams) else streams[-1])

    client = MagicMock()
    client.responses.create.side_effect = _create
    result = agent._run_codex_stream({"model": "test/model"}, client=client)
    return result, calls["n"]


def test_retryable_terminal_failure_retries_and_returns_second_attempt():
    agent = _agent()
    result, n = _run(agent, [_failed_frame(), _completed_frame("recovered")])

    assert n == 2, "a retryable terminal failure must be retried once in-stream"
    assert result.status == "completed"
    assert result.output_text == "recovered"


def test_retryable_terminal_failure_is_bounded():
    """The retry is bounded: a provider that keeps failing is returned, not looped on.

    This layer returns the failed response; the downstream normalizer raises on it. So the
    contract here is "one retry, then hand back the failure".
    """
    agent = _agent()
    result, n = _run(agent, [_failed_frame(), _failed_frame()])

    assert n == 2, "exactly one in-stream retry"
    assert result.status == "failed"


def test_incomplete_terminal_status_is_not_retried():
    agent = _agent()
    result, n = _run(agent, [_incomplete_frame()])

    assert n == 1, "status=incomplete is the truncation signal and must not consume a retry"
    assert result.status == "incomplete"


def test_non_retryable_failed_code_is_not_retried():
    agent = _agent()
    result, n = _run(agent, [_failed_frame(code="invalid_prompt", message="Request blocked.")])

    assert n == 1, "a permanent provider failure must not be retried"
    assert result.status == "failed"


def test_failed_attempt_text_does_not_leak_into_retry_accounting():
    agent = _agent()
    _run(agent, [_failed_frame(text="partial-bytes"), _completed_frame("final")])

    streamed = "".join(agent._codex_streamed_text_parts)
    assert "partial-bytes" not in streamed
    assert streamed == "final"


@pytest.mark.parametrize("final,expected", [
    (SimpleNamespace(status="failed", error=SimpleNamespace(code="server_error")), True),
    (SimpleNamespace(status="failed", error={"code": "server_error"}), True),
    (SimpleNamespace(status="failed", error=SimpleNamespace(code="SERVER_ERROR")), True),
    (SimpleNamespace(status="failed", error=SimpleNamespace(code="overloaded_error")), True),
    (SimpleNamespace(status="failed", error=SimpleNamespace(code="invalid_prompt")), False),
    (SimpleNamespace(status="failed", error=SimpleNamespace(code="")), False),
    (SimpleNamespace(status="failed", error=None), False),
    (SimpleNamespace(status="incomplete", error=SimpleNamespace(code="server_error")), False),
    (SimpleNamespace(status="completed", error=None), False),
])
def test_retryable_predicate(final, expected):
    from agent.codex_runtime import _is_retryable_terminal_failure

    assert _is_retryable_terminal_failure(final) is expected
