import time
from types import SimpleNamespace

import httpx
import pytest

from agent.codex_runtime import run_codex_stream

# Seconds the fake stream spends decoding between its first delta and its terminal frame.
_DECODE_GAP_S = 0.05
# A sleep can wake a few ms early on Windows, so the window lower bound stays loose.
_MIN_MEASURED_S = 0.04


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


def _completed_events():
    yield {
        "type": "response.created",
        "response": {"id": "fixture", "status": "in_progress"},
    }
    yield {"type": "response.output_text.delta", "delta": "Yes."}
    # Decode gap: the terminal frame lands after this stream's own decode seconds.
    time.sleep(_DECODE_GAP_S)
    yield {
        "type": "response.completed",
        "response": {"id": "fixture", "status": "completed"},
    }


def test_codex_stream_records_first_lifecycle_event_before_text(monkeypatch):
    agent = _agent()
    ticks = iter((100.0, 200.0, 300.0))
    monkeypatch.setattr("agent.codex_runtime.time.time", lambda: next(ticks))
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
    # The decode window opened at the first output delta and closed at the
    # terminal event — measured seconds, never an empty-default zero.
    decode_seconds = result.decode_seconds
    assert isinstance(decode_seconds, float)
    assert _MIN_MEASURED_S <= decode_seconds < 1.0
    # The measured window is handed to the usage recording path on success only.
    assert agent._last_api_decode_seconds == decode_seconds


def test_codex_reasoning_delta_opens_the_decode_window(monkeypatch):
    """Reasoning output is real model output: a reasoning delta starts the
    window even when no visible text delta ever arrives."""
    agent = _agent()

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
        # Decode gap: the terminal frame lands after this stream's own decode
        # seconds, measured from the reasoning delta that opened the window.
        time.sleep(_DECODE_GAP_S)
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
    assert _MIN_MEASURED_S <= decode_seconds < 1.0
    assert agent._last_api_decode_seconds == decode_seconds


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
