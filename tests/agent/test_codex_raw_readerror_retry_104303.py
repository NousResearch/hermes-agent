"""Regression coverage for raw Codex pre-stream ReadError recovery (#104303, #104452)."""

from types import SimpleNamespace

import httpx
import pytest

from agent import relay_llm
from agent.codex_runtime import run_codex_stream


class _Stream:
    def __init__(self, events=None, error_after=0, error=None):
        self._events = list(events or [])
        self._error_after = error_after
        self._error = error
        self._yielded = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._error is not None and self._yielded >= self._error_after:
            err = self._error
            self._error = None  # raise once
            raise err
        if self._yielded < len(self._events):
            event = self._events[self._yielded]
            self._yielded += 1
            return event
        raise StopIteration

    def close(self):
        self.closed = True


def _completed_stream():
    item = SimpleNamespace(
        type="message",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="Recovered.")],
    )
    return _Stream(
        [
            SimpleNamespace(type="response.output_item.done", item=item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="raw-read-retry"),
            ),
        ]
    )


def _agent(aborts):
    return SimpleNamespace(
        model="gpt-5-codex",
        provider="openai-codex",
        session_id="",
        is_subagent=False,
        _fallback_index=0,
        _interrupt_requested=False,
        _touch_activity=lambda _description: None,
        _client_log_context=lambda: "",
        _abort_request_openai_client=lambda _client, *, reason: aborts.append(reason),
    )


def _request():
    return {
        "model": "gpt-5-codex",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "Ping"}],
        "tools": None,
        "store": False,
    }


def test_raw_read_error_on_first_byte_retries_once_with_request_abort(monkeypatch):
    """A raw ReadError on reading the first SSE byte occurs after on_stream_created.

    Because no content or deltas were delivered, this is pre-stream. It must abort
    the request-local client and retry once, recovering cleanly.
    """
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        content=b'{"model":"gpt-5-codex"}',
    )
    calls = {"count": 0}
    aborts = []
    agent = _agent(aborts)

    def create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            # Stream handle exists, on_stream_created runs, but 1st byte fails
            return _Stream(error_after=0, error=httpx.ReadError("first byte EOF", request=request))
        return _completed_stream()

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    def relay_stream(api_kwargs, opener, **kwargs):
        stream = opener(api_kwargs)
        kwargs["on_stream_created"](stream)
        return stream

    monkeypatch.setattr(relay_llm, "stream", relay_stream)

    response = run_codex_stream(agent, _request(), client=client)

    assert calls["count"] == 2
    assert response.status == "completed"
    assert response.id == "raw-read-retry"
    assert aborts == ["codex_prestream_transport_retry"]


def test_raw_read_error_on_create_retries_once_with_request_abort(monkeypatch):
    """Raw ReadError raised directly by client.responses.create() before stream creation."""
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        content=b'{"model":"gpt-5-codex"}',
    )
    calls = {"count": 0}
    aborts = []
    agent = _agent(aborts)

    def create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadError("connect failed", request=request)
        return _completed_stream()

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    def relay_stream(api_kwargs, opener, **_kwargs):
        return opener(api_kwargs)

    monkeypatch.setattr(relay_llm, "stream", relay_stream)

    response = run_codex_stream(agent, _request(), client=client)

    assert calls["count"] == 2
    assert response.status == "completed"
    assert response.id == "raw-read-retry"
    assert aborts == ["codex_prestream_transport_retry"]


def test_raw_read_error_after_events_delivered_is_not_retried(monkeypatch):
    """A raw ReadError mid-stream after events were delivered must NOT retry.

    Replaying an inference after partial output was received could duplicate
    output and bill twice.
    """
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        content=b'{"model":"gpt-5-codex"}',
    )
    calls = {"count": 0}
    aborts = []
    agent = _agent(aborts)

    def create(**_kwargs):
        calls["count"] += 1
        # Yields one delta event, then raises ReadError on the next read
        delta_event = SimpleNamespace(type="response.output_text.delta", delta="Partial output")
        return _Stream(events=[delta_event], error_after=1, error=httpx.ReadError("mid-stream receive failed", request=request))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    def relay_stream(api_kwargs, opener, **kwargs):
        stream = opener(api_kwargs)
        kwargs["on_stream_created"](stream)
        return stream

    monkeypatch.setattr(relay_llm, "stream", relay_stream)

    with pytest.raises(httpx.ReadError, match="mid-stream receive failed"):
        run_codex_stream(agent, _request(), client=client)

    assert calls["count"] == 1
    assert aborts == []


def test_raw_read_error_prestream_exhaustion_raises(monkeypatch):
    """When pre-stream ReadError persists across max_stream_retries, it raises."""
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        content=b'{"model":"gpt-5-codex"}',
    )
    calls = {"count": 0}
    aborts = []
    agent = _agent(aborts)

    def create(**_kwargs):
        calls["count"] += 1
        return _Stream(error_after=0, error=httpx.ReadError("repeated pre-stream failure", request=request))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    def relay_stream(api_kwargs, opener, **kwargs):
        stream = opener(api_kwargs)
        kwargs["on_stream_created"](stream)
        return stream

    monkeypatch.setattr(relay_llm, "stream", relay_stream)

    with pytest.raises(httpx.ReadError, match="repeated pre-stream failure"):
        run_codex_stream(agent, _request(), client=client)

    assert calls["count"] == 2
    assert aborts == ["codex_prestream_transport_retry"]


def test_raw_read_error_through_real_relay_pipeline():
    """Exercise the REAL relay_llm.stream pipeline without monkeypatching relay.

    Validates that the real ManagedLlmStream unmanaged lifecycle and on_stream_created
    do not block pre-stream raw ReadError retry.
    """
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        content=b'{"model":"gpt-5-codex"}',
    )
    calls = {"count": 0}
    aborts = []
    agent = _agent(aborts)

    def create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _Stream(error_after=0, error=httpx.ReadError("first SSE byte EOF", request=request))
        return _completed_stream()

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    # relay_llm.stream is NOT monkeypatched; runs the real Relay unmanaged code
    response = run_codex_stream(agent, _request(), client=client)

    assert calls["count"] == 2
    assert response.status == "completed"
    assert response.id == "raw-read-retry"
    assert aborts == ["codex_prestream_transport_retry"]


def test_three_consecutive_prestream_recovery_cycles_no_bottleneck():
    """3-attempt / 3-turn validation: consecutive pre-stream failures recover without bottleneck.

    Ensures that multiple recovery cycles clean up sockets, abort request clients
    atomically, leave no state leakage, and do not hang or deadlock.
    """
    request = httpx.Request(
        "POST",
        "https://chatgpt.com/backend-api/codex/responses",
        content=b'{"model":"gpt-5-codex"}',
    )
    calls = {"count": 0}
    aborts = []
    agent = _agent(aborts)

    def create(**_kwargs):
        calls["count"] += 1
        # Odd calls fail on 1st byte; even calls succeed
        if calls["count"] % 2 == 1:
            return _Stream(error_after=0, error=httpx.ReadError(f"cycle failure {calls['count']}", request=request))
        return _completed_stream()

    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    # Run 3 consecutive turns
    for cycle in range(3):
        resp = run_codex_stream(agent, _request(), client=client)
        assert resp.status == "completed"
        assert resp.id == "raw-read-retry"

    assert calls["count"] == 6  # 3 cycles * 2 physical attempts each
    assert aborts == ["codex_prestream_transport_retry"] * 3
