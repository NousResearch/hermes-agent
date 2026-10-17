import pytest
from types import SimpleNamespace

import httpx
import pytest

from agent.codex_runtime import run_codex_stream


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="",
        provider="openai-codex",
        model="timing-fixture",
        _interrupt_requested=False,
        _last_api_first_chunk_at=None,
        _touch_activity=lambda *_: None,
        _fire_stream_delta=lambda *_: None,
        _fire_reasoning_delta=lambda *_: None,
        _client_log_context=lambda: "",
    )


def _mock_monotonic(monkeypatch, *stamps):
    """Deterministic clock: each monotonic() call hands out the next scripted stamp.

    Stamps land only at the decode-stamp sites (first real-data delta, clean
    response.completed frame), so the scripted values measure exact windows —
    no real sleeps, no OS clock-tick races.
    """
    stamps_iter = iter(stamps)
    # The test replaced the process-wide time.monotonic, so ambient users of the
    # clock keep calling it after the test body: Windows proactor shutdown calls
    # time.monotonic() while the conftest closes the loop during fixture
    # teardown. An exhausted queue must not raise there — the clock repeats its
    # last stamp instead of raising StopIteration.
    monkeypatch.setattr(
        "agent.codex_runtime.time.monotonic",
        lambda: next(stamps_iter, stamps[-1] if stamps else 0.0),
    )


def _completed_events():
    yield {
        "type": "response.created",
        "response": {"id": "fixture", "status": "in_progress"},
    }
    yield {"type": "response.output_text.delta", "delta": "Yes."}
    yield {
        "type": "response.completed",
        "response": {"id": "fixture", "status": "completed"},
    }


def test_codex_stream_records_first_lifecycle_event_before_text(monkeypatch):
    agent = _agent()
    ticks = iter((100.0, 200.0, 300.0))
    monkeypatch.setattr("agent.codex_runtime.time.time", lambda: next(ticks))
    # The second (successful) attempt stamps its own decode window: first output
    # delta at 10.0, the completed frame at 100.5 — never the failed attempt's wait.
    _mock_monotonic(monkeypatch, 10.0, 100.5)
    attempts = 0

    def create(**_):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("transient connect failure")
        return _completed_events()

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Say Yes."}, client=client
    )

    assert result.output_text == "Yes."
    assert result.status == "completed"
    assert attempts == 2
    assert agent._last_api_first_chunk_at == 100.0
    # The decode window opened at the second attempt's first output delta and
    # closed at its terminal event — measured seconds, never an empty-default zero.
    decode_seconds = result.decode_seconds
    assert isinstance(decode_seconds, float)
    assert decode_seconds == 90.5
    # The measured window is handed to the usage recording path on success only.
    assert agent._last_api_decode_seconds == 90.5


def test_codex_reasoning_delta_opens_the_decode_window(monkeypatch):
    """Reasoning output is real model output: a reasoning delta starts the
    window even when no visible text delta ever arrives."""
    agent = _agent()
    _mock_monotonic(monkeypatch, 10.0, 100.5)

    def events():
        yield {
            "type": "response.created",
            "response": {"id": "fixture", "status": "in_progress"},
        }
        yield {
            "type": "response.output_item.added",
            "item": {"type": "reasoning", "id": "r1"},
        }
        yield {
            "type": "response.reasoning_text.delta",
            "item_id": "r1",
            "index": 0,
            "offset": 0,
            "delta": "thinking",
        }
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "r1",
                "summary": [{"type": "summary_text", "text": "thinking"}],
            },
        }
        yield {
            "type": "response.completed",
            "response": {"id": "fixture", "status": "completed"},
        }

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Think."}, client=client
    )

    decode_seconds = result.decode_seconds
    assert isinstance(decode_seconds, float)
    assert decode_seconds == 90.5
    assert agent._last_api_decode_seconds == 90.5


def test_codex_incomplete_terminal_leaves_the_window_unmeasured(monkeypatch):
    """An incomplete terminal ends the call without a clean finish: tokens that
    did arrive keep the window unmeasured — a partial call is not a speed."""
    agent = _agent()
    _mock_monotonic(monkeypatch, 10.0, 100.5)

    def events():
        yield {
            "type": "response.created",
            "response": {"id": "fixture", "status": "in_progress"},
        }
        yield {"type": "response.output_text.delta", "delta": "Half an answer"}
        yield {
            "type": "response.incomplete",
            "response": {
                "id": "fixture",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        }

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Long."}, client=client
    )

    assert result.status == "incomplete"
    assert result.decode_seconds is None
    assert agent._last_api_decode_seconds is None


def test_codex_failed_terminal_leaves_the_window_unmeasured(monkeypatch):
    """A failed terminal is not a clean completion: even with tokens delivered,
    the window stays unmeasured."""
    agent = _agent()
    _mock_monotonic(monkeypatch, 10.0, 100.5)

    def events():
        yield {
            "type": "response.created",
            "response": {"id": "fixture", "status": "in_progress"},
        }
        yield {"type": "response.output_text.delta", "delta": "Cut answer"}
        yield {
            "type": "response.failed",
            "response": {
                "id": "fixture",
                "status": "failed",
                "error": {"type": "server_error", "message": "boom"},
            },
        }

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Crash."}, client=client
    )

    assert result.status == "failed"
    assert result.decode_seconds is None
    assert agent._last_api_decode_seconds is None


def test_codex_function_call_announcement_without_arguments_does_not_open_the_window():
    """An announced function call may carry no generated payload yet: the bare
    announcement must not open the decode window — only argument data that
    reached the wire counts, so this stream stays unmeasured."""
    agent = _agent()

    def events():
        yield {
            "type": "response.created",
            "response": {"id": "fixture", "status": "in_progress"},
        }
        yield {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "f1", "call_id": "c1", "name": "tool_x", "arguments": ""},
        }
        yield {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "id": "f1", "call_id": "c1", "name": "tool_x", "arguments": ""},
        }
        yield {
            "type": "response.completed",
            "response": {"id": "fixture", "status": "completed"},
        }

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Tool."}, client=client
    )

    assert result.status == "completed"
    assert result.decode_seconds is None
    assert agent._last_api_decode_seconds is None


def test_codex_function_call_arguments_delta_opens_the_window(monkeypatch):
    """Real tool-call arguments arriving on the wire open the window: the
    announcement alone does not — the argument delta's arrival does."""
    agent = _agent()
    _mock_monotonic(monkeypatch, 10.0, 100.5)

    def events():
        yield {
            "type": "response.created",
            "response": {"id": "fixture", "status": "in_progress"},
        }
        yield {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "f1", "call_id": "c1", "name": "tool_x", "arguments": ""},
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "f1",
            "index": 0,
            "delta": '{"x": 1}',
        }
        yield {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "id": "f1", "call_id": "c1", "name": "tool_x", "arguments": '{"x": 1}'},
        }
        yield {
            "type": "response.completed",
            "response": {"id": "fixture", "status": "completed"},
        }

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Tool."}, client=client
    )

    assert result.decode_seconds == 90.5
    assert agent._last_api_decode_seconds == 90.5


def test_codex_stream_without_output_keeps_decode_window_unmeasured(monkeypatch):
    """A stream that ends without any output delta measured nothing: the
    window stays None rather than a fabricated near-instant speed."""
    agent = _agent()

    def events():
        yield {
            "type": "response.created",
            "response": {"id": "fixture", "status": "in_progress"},
        }
        yield {
            "type": "response.completed",
            "response": {"id": "fixture", "status": "incomplete"},
        }

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    result = run_codex_stream(
        agent, {"model": "timing-fixture", "input": "Quiet."}, client=client
    )

    assert result.decode_seconds is None
    assert agent._last_api_decode_seconds is None


@pytest.mark.parametrize("retire_before_event", [False, True])
def test_codex_stream_without_accepted_event_keeps_timing_unset(retire_before_event):
    agent = _agent()
    request_token = object()
    agent._active_codex_stream_request_token = request_token

    def events():
        if retire_before_event:
            agent._active_codex_stream_request_token = object()
            yield {"type": "response.created"}
        return
        yield

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: events()))

    with pytest.raises((RuntimeError, TimeoutError)):
        run_codex_stream(agent, {"model": "timing-fixture"}, client=client)

    assert agent._last_api_first_chunk_at is None
