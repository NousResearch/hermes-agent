"""Spend-gate enforcement at the Codex Responses chokepoint.

These cover the integration added in ``agent/codex_runtime.run_codex_stream``:

* when the process-wide guard DENIES, the stream raises
  ``CodexSpendCapError`` and never calls ``client.responses.create``;
* when ALLOWED, the call proceeds and the guard records the completed
  turn's tokens (snapshot reflects 1 call + prompt+output tokens).

The autouse ``_isolate_codex_spend_ledger`` fixture (tests/conftest.py)
already points the guard at a per-test ledger and resets the singleton,
so these tests never touch the real ``~/.hermes/codex_spend.json``.
"""

from types import SimpleNamespace

import pytest

import agent.codex_runtime as codex_runtime
from agent.codex_spend_guard import (
    CodexSpendCapError,
    Limits,
    get_codex_spend_guard,
    reset_codex_spend_guard_for_test,
)


class _FakeAgent:
    """Minimal stand-in for AIAgent for the run_codex_stream call path."""

    def __init__(self, client):
        self._client = client
        self._interrupt_requested = False
        self._codex_streamed_text_parts = []
        self._codex_stream_last_event_ts = 0.0

    def _ensure_primary_openai_client(self, reason=None):
        return self._client

    def _fire_stream_delta(self, text):
        pass

    def _fire_reasoning_delta(self, text):
        pass

    def _touch_activity(self, *_a, **_k):
        pass

    def _client_log_context(self):
        return "fake-client"


def _completed_event_stream(usage):
    """Yield a minimal Codex SSE event sequence that ``run_codex_stream``
    assembles into a completed ``final`` response carrying ``usage``.

    The stream is an iterable (NOT a concrete ``.output`` object), so it goes
    through the real ``_consume_codex_event_stream`` assembly path — which is
    where the spend-gate token recording lives.
    """
    message_item = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="hi")],
    )
    return [
        SimpleNamespace(type="response.output_text.delta", delta="hi"),
        SimpleNamespace(type="response.output_item.done", item=message_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", id="resp_1", usage=usage),
        ),
    ]


class _RecordingResponses:
    def __init__(self, events):
        self._events = events
        self.create_calls = 0

    def create(self, **_kwargs):
        self.create_calls += 1
        # Return a fresh iterator each call.
        return iter(list(self._events))


class _RecordingClient:
    def __init__(self, events):
        self.responses = _RecordingResponses(events)


def _seed_singleton_over_ceiling(monkeypatch):
    """Force the process-wide guard to a limits=1/hour config and burn the
    single allowed call so the NEXT reserve() denies."""
    reset_codex_spend_guard_for_test()
    guard = get_codex_spend_guard()
    guard.limits = Limits(
        max_calls_per_hour=1, max_calls_per_day=100, max_tokens_per_day=10**9
    )
    # Burn the one allowed call so the gate's reserve() will deny.
    first = guard.reserve()
    assert first.allowed is True
    return guard


def test_run_codex_stream_denies_when_guard_over_ceiling(monkeypatch):
    _seed_singleton_over_ceiling(monkeypatch)
    usage = SimpleNamespace(prompt_tokens=100, output_tokens=50)
    client = _RecordingClient(_completed_event_stream(usage))
    agent = _FakeAgent(client)

    with pytest.raises(CodexSpendCapError) as excinfo:
        codex_runtime.run_codex_stream(agent, {"model": "gpt-5-codex"}, client=client)

    assert excinfo.value.reason == "calls_per_hour"
    # The chokepoint must short-circuit BEFORE touching the API.
    assert client.responses.create_calls == 0


def test_run_codex_stream_proceeds_and_records_tokens_when_allowed(monkeypatch):
    reset_codex_spend_guard_for_test()
    guard = get_codex_spend_guard()  # default hard ceilings → allowed
    usage = SimpleNamespace(prompt_tokens=100, output_tokens=50)
    client = _RecordingClient(_completed_event_stream(usage))
    agent = _FakeAgent(client)

    result = codex_runtime.run_codex_stream(
        agent, {"model": "gpt-5-codex"}, client=client
    )

    assert result.status == "completed"
    assert client.responses.create_calls == 1

    snap = guard.snapshot()
    assert snap["calls_last_hour"] == 1
    assert snap["tokens_last_day"] == 150


def test_run_codex_stream_records_completion_tokens_fallback(monkeypatch):
    """When usage exposes ``completion_tokens`` (not ``output_tokens``) it is
    still recorded via the getattr fallback."""
    reset_codex_spend_guard_for_test()
    guard = get_codex_spend_guard()
    usage = SimpleNamespace(prompt_tokens=200, completion_tokens=25)
    client = _RecordingClient(_completed_event_stream(usage))
    agent = _FakeAgent(client)

    codex_runtime.run_codex_stream(agent, {"model": "gpt-5-codex"}, client=client)

    assert guard.snapshot()["tokens_last_day"] == 225
