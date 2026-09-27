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
