"""Codex Responses stream drops recover the way the chat_completions wire does.

A provider (or a proxy in front of it) that cuts one physical Responses stream is usually
healthy on the next request, but ``run_codex_stream`` failed the whole turn on the first
drop unless the SDK surfaced it as an ``httpx`` transport error. Two shapes it did not
recover, both of which end a turn with a "connection dropped" card:

* a stream that reaches EOF without its terminal frame and without producing any output
  (``RuntimeError: Codex Responses stream did not emit a terminal response``), and
* a Responses ``type=error`` SSE frame reporting a transient upstream failure — the same
  class ``_handle_stream_error`` retries on the chat wire.

Both are safe to replay while nothing visible was delivered (or only a not-yet-executed
tool call was in flight), and neither may be replayed after visible text (the re-streamed
preamble would duplicate it) or when the classifier calls the failure deterministic.
"""

import sys
import types
from types import SimpleNamespace

import httpx
import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent import chat_completion_helpers
from agent.codex_runtime import run_codex_stream


@pytest.fixture(autouse=True)
def _no_wall_clock_backoff(monkeypatch):
    """Record the backoff instead of sleeping through it (retry ordering is what matters)."""
    waits: list[float] = []
    monkeypatch.setattr(chat_completion_helpers, "_wait_stream_retry_backoff",
                        lambda agent, delay: waits.append(delay))
    return waits


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", dict)


def _build_agent(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    agent = run_agent.AIAgent(
        model="step-5-preview",
        base_url="https://api.stepfun.com/step_plan/v1",
        api_key="step-key",
        api_mode="codex_responses",
        quiet_mode=True,
        max_iterations=4,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


class _FakeStream:
    """Iterable-only ``responses.create(stream=True)`` result, like the wire's SSE body."""

    def __init__(self, events, *, fail_after=None, error=None):
        self._events = list(events)
        self._fail_after = fail_after
        self._error = error
        self.closed = False

    def __iter__(self):
        if self._fail_after is None:
            return iter(self._events)
        return self._drop_after()

    def _drop_after(self):
        for index, event in enumerate(self._events):
            if index == self._fail_after:
                raise self._error
            yield event
        raise self._error

    def close(self):
        self.closed = True


def _request_kwargs():
    return {
        "model": "step-5-preview",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "Ping"}],
        "tools": None,
        "store": False,
    }


def _lifecycle_only():
    return [SimpleNamespace(type="response.created", response=SimpleNamespace(id="resp_1"))]


def _text_events(*deltas):
    return [
        SimpleNamespace(type="response.output_item.added",
                        item=SimpleNamespace(type="message", role="assistant", phase=None, id="m1")),
        *(SimpleNamespace(type="response.output_text.delta", delta=text) for text in deltas),
    ]


def _completed(text="Recovered."):
    return [
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="message", status="completed", id="m2",
                                             content=[SimpleNamespace(type="output_text", text=text)])),
        SimpleNamespace(type="response.completed",
                        response=SimpleNamespace(status="completed", usage=None, id="resp_ok")),
    ]


def _error_frame(message, code=None):
    return {"type": "error", "error": {"message": message, "code": code}}


def _first_text(response):
    """Text of the first message item in an assembled Responses object."""
    return response.output[0].content[0].text


def _client(create):
    return SimpleNamespace(responses=SimpleNamespace(create=create))


def test_terminal_less_stream_retries_and_recovers_before_any_output(monkeypatch, _no_wall_clock_backoff):
    calls = {"count": 0}

    def _create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeStream(_lifecycle_only())  # clean EOF, no output, no terminal frame
        return _FakeStream(_completed())

    agent = _build_agent(monkeypatch)
    agent.client = _client(_create)

    response = run_codex_stream(agent, _request_kwargs())

    assert calls["count"] == 2
    assert response.status == "completed"
    assert _first_text(response) == "Recovered."
    # Every physical retry goes through the chat wire's interruptible backoff hook.
    assert _no_wall_clock_backoff == [0.0]


def test_terminal_less_stream_exhaustion_raises_after_the_shared_retry_budget(monkeypatch, _no_wall_clock_backoff):
    calls = {"count": 0}

    def _create(**_kwargs):
        calls["count"] += 1
        return _FakeStream(_lifecycle_only())

    agent = _build_agent(monkeypatch)
    agent.client = _client(_create)

    with pytest.raises(RuntimeError, match="did not emit a terminal response"):
        run_codex_stream(agent, _request_kwargs())

    # HERMES_STREAM_RETRIES defaults to 2: three physical attempts, like the chat wire.
    assert calls["count"] == 3


def test_hermes_stream_retries_bounds_the_codex_stream_budget(monkeypatch, _no_wall_clock_backoff):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")
    calls = {"count": 0}

    def _create(**_kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            return _FakeStream(_lifecycle_only())
        return _FakeStream(_completed())

    agent = _build_agent(monkeypatch)
    agent.client = _client(_create)

    response = run_codex_stream(agent, _request_kwargs())

    assert calls["count"] == 3
    assert response.status == "completed"


def test_transient_error_frame_retries_before_any_delivery(monkeypatch, _no_wall_clock_backoff):
    calls = {"count": 0}

    def _create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeStream([*_lifecycle_only(), _error_frame("Connection lost", "server_error")])
        return _FakeStream(_completed())

    agent = _build_agent(monkeypatch)
    agent.client = _client(_create)

    response = run_codex_stream(agent, _request_kwargs())

    assert calls["count"] == 2
    assert response.status == "completed"


def test_error_frame_after_visible_text_is_not_retried(monkeypatch, _no_wall_clock_backoff):
    calls = {"count": 0}

    def _create(**_kwargs):
        calls["count"] += 1
        return _FakeStream([*_text_events("Half a "), _error_frame("Connection lost", "server_error")])

    agent = _build_agent(monkeypatch)
    agent.client = _client(_create)

    with pytest.raises(Exception, match="Connection lost"):
        run_codex_stream(agent, _request_kwargs())

    assert calls["count"] == 1


def test_deterministic_error_frame_is_not_retried(monkeypatch, _no_wall_clock_backoff):
    calls = {"count": 0}

    def _create(**_kwargs):
        calls["count"] += 1
        return _FakeStream([_error_frame("Invalid API key", "invalid_api_key")])

    agent = _build_agent(monkeypatch)
    agent.client = _client(_create)

    with pytest.raises(Exception, match="Invalid API key"):
        run_codex_stream(agent, _request_kwargs())

    assert calls["count"] == 1


def test_mid_tool_call_drop_retries_after_text_with_a_reconnect_marker(monkeypatch, _no_wall_clock_backoff):
    calls = {"count": 0}
    announced_tool = SimpleNamespace(type="response.output_item.added",
                                     item=SimpleNamespace(type="function_call", id="fc_1", call_id="call_1",
                                                          name="terminal", arguments=""))

    def _create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeStream(
                [*_text_events("Let me check. "), announced_tool],
                # The tool item is announced (never executed) before the transport drops.
                fail_after=3, error=httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
            )
        return _FakeStream(_completed("Done."))

    agent = _build_agent(monkeypatch)
    agent._warning_presentation_enabled = lambda: True
    agent.client = _client(_create)
    streamed: list[str] = []
    agent._fire_stream_delta = streamed.append

    response = run_codex_stream(agent, _request_kwargs())

    assert calls["count"] == 2
    assert _first_text(response) == "Done."
    assert any("reconnecting" in text for text in streamed)
