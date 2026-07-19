"""Unit coverage for the opt-in generic Codex Responses WebSocket transport."""

import json
from types import SimpleNamespace

import pytest


def test_normalize_responses_transport_accepts_only_known_values():
    from agent.codex_responses_ws_transport import normalize_responses_transport

    assert normalize_responses_transport("websocket") == "websocket"
    assert normalize_responses_transport(" AUTO ") == "auto"
    assert normalize_responses_transport("sse") == "sse"
    assert normalize_responses_transport("websocket-cached") == "sse"
    assert normalize_responses_transport(None) == "sse"


def test_generic_ws_eligibility_is_limited_to_named_custom_codex_providers():
    from agent.codex_responses_ws_transport import is_generic_codex_ws_eligible

    assert is_generic_codex_ws_eligible(
        provider="custom:sub2api",
        base_url="https://relay.example.com/v1",
        api_mode="codex_responses",
    )
    assert not is_generic_codex_ws_eligible(
        provider="custom",
        base_url="https://relay.example.com/v1",
        api_mode="codex_responses",
    )
    assert not is_generic_codex_ws_eligible(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
    )
    assert not is_generic_codex_ws_eligible(
        provider="custom:sub2api",
        base_url="https://chatgpt.com/v1",
        api_mode="codex_responses",
    )
    assert not is_generic_codex_ws_eligible(
        provider="custom:sub2api",
        base_url="https://relay.example.com/v1",
        api_mode="chat_completions",
    )
    assert not is_generic_codex_ws_eligible(
        provider="custom:sub2api",
        base_url="http://[malformed",
        api_mode="codex_responses",
    )


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://relay.example.com/v1", "wss://relay.example.com/v1/responses"),
        ("http://relay.example.com/v1/", "ws://relay.example.com/v1/responses"),
        ("https://relay.example.com/responses", "wss://relay.example.com/responses"),
        ("http://relay.example.com/api", "ws://relay.example.com/api/responses"),
    ],
)
def test_resolve_responses_ws_url_derives_endpoint(base_url, expected):
    from agent.codex_responses_ws_transport import resolve_responses_ws_url

    assert resolve_responses_ws_url(base_url) == expected
    assert (
        resolve_responses_ws_url(base_url, "wss://override.example/responses")
        == "wss://override.example/responses"
    )


def test_build_ws_wire_body_removes_sdk_only_fields_and_merges_extra_body():
    from agent.codex_responses_ws_transport import build_ws_wire_body

    body = build_ws_wire_body(
        {
            "model": "gpt-5",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
            "timeout": 10,
            "extra_headers": {"X-Relay": "value"},
            "extra_query": {"version": "1"},
            "extra_body": {"reasoning": {"effort": "low"}, "stream": False},
        }
    )

    assert body == {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "low"},
    }


def test_build_headers_drops_openai_omit_sentinels():
    from agent.codex_responses_ws_transport import _build_headers

    class _Omit:
        __module__ = "openai"

        def __str__(self) -> str:
            return "<openai.Omit object at 0xdeadbeef>"

    class _Client:
        default_headers = {
            "Accept": "application/json",
            "Authorization": "Bearer from-default",
            "OpenAI-Organization": _Omit(),
            "OpenAI-Project": _Omit(),
            "X-Stainless-Lang": "python",
        }
        _custom_headers = {
            "X-Custom-From-SDK": "yes",
            "OpenAI-Organization": _Omit(),
        }
        api_key = "sk-should-not-override-existing-auth"

    headers = _build_headers(
        api_kwargs={"extra_headers": {"X-Relay": "1", "OpenAI-Project": _Omit()}},
        client=_Client(),
        api_key="sk-unused",
        headers={"X-Explicit": "ok"},
    )

    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer from-default"
    assert headers["X-Custom-From-SDK"] == "yes"
    assert headers["X-Relay"] == "1"
    assert headers["X-Explicit"] == "ok"
    assert "OpenAI-Organization" not in headers
    assert "OpenAI-Project" not in headers
    assert not any("Omit object" in str(v) for v in headers.values())


def test_build_generic_ws_identity_includes_ws_url_and_transport():
    from agent.codex_responses_ws_transport import build_generic_ws_identity

    a = build_generic_ws_identity(
        session_id="s1",
        transport_provider="custom:sub2api",
        base_url="https://relay.example.com/v1",
        model="gpt-5",
        responses_ws_url=None,
        transport="auto",
    )
    b = build_generic_ws_identity(
        session_id="s1",
        transport_provider="custom:sub2api",
        base_url="https://relay.example.com/v1",
        model="gpt-5",
        responses_ws_url="wss://relay.example.com/ws/responses",
        transport="auto",
    )
    assert a != b


class _FakeSocket:
    def __init__(self, frames=(), send_error=None, recv_timeouts=0):
        self._frames = list(frames)
        self._send_error = send_error
        self._recv_timeouts = recv_timeouts
        self.sent = []
        self.closed = False
        self.recv_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def send(self, payload):
        self.sent.append(payload)
        if self._send_error is not None:
            raise self._send_error

    def recv(self, timeout=None):
        self.recv_calls += 1
        if self._recv_timeouts > 0:
            self._recv_timeouts -= 1
            raise TimeoutError("poll idle")
        if not self._frames:
            raise TimeoutError("no more frames")
        return self._frames.pop(0)

    def close(self):
        self.closed = True


def test_generic_ws_stream_sends_response_create_and_reuses_event_consumer(monkeypatch):
    import agent.codex_responses_ws_transport as transport

    socket = _FakeSocket(
        [
            json.dumps({"type": "response.output_text.delta", "delta": "hello"}),
            json.dumps({"type": "response.done", "response": {"status": "completed"}}),
        ]
    )
    monkeypatch.setattr(transport, "_connect_websocket", lambda *_args, **_kwargs: socket)
    collected = []

    def consume(events, _unused_client):
        collected.extend(events)
        return SimpleNamespace(status="completed", output_text="hello")

    result = transport.run_generic_codex_ws_stream(
        api_kwargs={"model": "gpt-5", "input": "hi"},
        api_key="test-key",
        provider="custom:sub2api",
        base_url="https://relay.example.com/v1",
        session_id="session-1",
        transport="websocket",
        collect_events=consume,
        interrupted=lambda: False,
    )

    assert json.loads(socket.sent[0]) == {
        "type": "response.create",
        "model": "gpt-5",
        "input": "hi",
    }
    assert [event.type for event in collected] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert result.output_text == "hello"


def test_ws_failure_after_send_is_not_replay_safe(monkeypatch):
    import agent.codex_responses_ws_transport as transport

    socket = _FakeSocket(send_error=OSError("connection dropped"))
    monkeypatch.setattr(transport, "_connect_websocket", lambda *_args, **_kwargs: socket)

    with pytest.raises(transport.GenericWsStartedError):
        transport.run_generic_codex_ws_stream(
            api_kwargs={"model": "gpt-5", "input": "hi"},
            api_key="test-key",
            provider="custom:sub2api",
            base_url="https://relay.example.com/v1",
            session_id="session-1",
            transport="websocket",
            collect_events=lambda events, _client: list(events),
            interrupted=lambda: False,
        )


def test_ws_rejected_error_is_structured(monkeypatch):
    import agent.codex_responses_ws_transport as transport

    socket = _FakeSocket(
        [
            json.dumps(
                {
                    "type": "error",
                    "status_code": 400,
                    "error": {"message": "bad request"},
                }
            )
        ]
    )
    monkeypatch.setattr(transport, "_connect_websocket", lambda *_args, **_kwargs: socket)

    with pytest.raises(transport.GenericWsRejectedError) as excinfo:
        transport.run_generic_codex_ws_stream(
            api_kwargs={"model": "gpt-5", "input": "hi"},
            api_key="test-key",
            provider="custom:sub2api",
            base_url="https://relay.example.com/v1",
            session_id="session-1",
            transport="websocket",
            collect_events=lambda events, _client: list(events),
            interrupted=lambda: False,
        )
    assert excinfo.value.status_code == 400
    assert "bad request" in str(excinfo.value)


def test_ws_cancelled_terminal_ends_stream(monkeypatch):
    import agent.codex_responses_ws_transport as transport

    socket = _FakeSocket(
        [
            json.dumps({"type": "response.done", "response": {"status": "canceled"}}),
        ]
    )
    monkeypatch.setattr(transport, "_connect_websocket", lambda *_args, **_kwargs: socket)
    collected = []

    def consume(events, _unused):
        collected.extend(events)
        return SimpleNamespace(status="cancelled")

    result = transport.run_generic_codex_ws_stream(
        api_kwargs={"model": "gpt-5", "input": "hi"},
        api_key="test-key",
        provider="custom:sub2api",
        base_url="https://relay.example.com/v1",
        session_id="session-1",
        transport="websocket",
        collect_events=consume,
        interrupted=lambda: False,
    )
    assert [event.type for event in collected] == ["response.cancelled"]
    assert result.status == "cancelled"


def test_run_codex_stream_ws_cancelled_uses_shared_consumer(monkeypatch):
    """Cancelled WS frames must terminate via the production collector, not a fake one."""
    import agent.codex_responses_ws_transport as transport
    from agent.codex_runtime import run_codex_stream

    socket = _FakeSocket(
        [
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_ws_cancelled",
                        "status": "canceled",
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr(transport, "_connect_websocket", lambda *_args, **_kwargs: socket)
    sse_calls = []

    def create(**kwargs):
        sse_calls.append(kwargs)
        return iter([])

    agent = SimpleNamespace(
        responses_transport="websocket",
        responses_transport_provider="custom:sub2api",
        responses_ws_url=None,
        _generic_ws_auto_disabled_for=None,
        provider="custom",
        api_mode="codex_responses",
        base_url="https://relay.example.com/v1",
        api_key="test-key",
        session_id="session-1",
        model="gpt-5",
        _client_kwargs={},
        _interrupt_requested=False,
        _codex_streamed_text_parts=[],
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _fire_streamed_codex_commentary=lambda _text: None,
        _touch_activity=lambda _message: None,
        _client_log_context=lambda: "test",
        interim_assistant_callback=None,
        show_commentary=True,
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    response = run_codex_stream(
        agent,
        {"model": "gpt-5", "input": "hello"},
        client=client,
    )
    assert response.status == "cancelled"
    assert response.id == "resp_ws_cancelled"
    assert response.output == []
    assert sse_calls == []


def test_ws_idle_timeout_raises_started_error(monkeypatch):
    import agent.codex_responses_ws_transport as transport

    socket = _FakeSocket(frames=[], recv_timeouts=100)
    monkeypatch.setattr(transport, "_connect_websocket", lambda *_args, **_kwargs: socket)

    with pytest.raises(transport.GenericWsStartedError):
        transport.run_generic_codex_ws_stream(
            api_kwargs={"model": "gpt-5", "input": "hi"},
            api_key="test-key",
            provider="custom:sub2api",
            base_url="https://relay.example.com/v1",
            session_id="session-1",
            transport="websocket",
            collect_events=lambda events, _client: list(events),
            interrupted=lambda: False,
            idle_timeout=0.01,
            recv_poll_timeout=0.001,
        )


def test_explicit_websocket_mode_does_not_fallback_to_sse(monkeypatch):
    import agent.codex_responses_ws_transport as transport
    from agent.codex_runtime import run_codex_stream

    def fail_before_send(**_kwargs):
        raise transport.GenericWsNotStartedError("upgrade unavailable")

    monkeypatch.setattr(transport, "run_generic_codex_ws_stream", fail_before_send)
    sse_calls = []

    def create(**kwargs):
        sse_calls.append(kwargs)
        return iter([])

    agent = SimpleNamespace(
        responses_transport="websocket",
        responses_transport_provider="custom:sub2api",
        responses_ws_url=None,
        _generic_ws_auto_disabled_for=None,
        provider="custom",
        api_mode="codex_responses",
        base_url="https://relay.example.com/v1",
        api_key="test-key",
        session_id="session-1",
        model="gpt-5",
        _client_kwargs={},
        _interrupt_requested=False,
        _codex_streamed_text_parts=[],
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _fire_streamed_codex_commentary=lambda _text: None,
        _touch_activity=lambda _message: None,
        _client_log_context=lambda: "test",
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with pytest.raises(transport.GenericWsNotStartedError):
        run_codex_stream(agent, {"model": "gpt-5", "input": "hello"}, client=client)
    assert sse_calls == []


def test_auto_transport_sticks_to_sse_after_pre_send_ws_failure(monkeypatch):
    import agent.codex_responses_ws_transport as transport
    from agent.codex_runtime import run_codex_stream

    ws_calls = []

    def fail_before_send(**_kwargs):
        ws_calls.append(True)
        raise transport.GenericWsNotStartedError("upgrade unavailable")

    monkeypatch.setattr(transport, "run_generic_codex_ws_stream", fail_before_send)
    output_item = SimpleNamespace(
        type="message",
        status="completed",
        content=[SimpleNamespace(type="output_text", text="SSE fallback")],
    )
    sse_calls = []

    def create(**kwargs):
        sse_calls.append(kwargs)
        return iter(
            [
                SimpleNamespace(type="response.output_item.done", item=output_item),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(status="completed"),
                ),
            ]
        )

    agent = SimpleNamespace(
        responses_transport="auto",
        responses_transport_provider="custom:sub2api",
        responses_ws_url=None,
        _generic_ws_auto_disabled_for=None,
        provider="custom",
        api_mode="codex_responses",
        base_url="https://relay.example.com/v1",
        api_key="test-key",
        session_id="session-1",
        model="gpt-5",
        _client_kwargs={},
        _interrupt_requested=False,
        _codex_streamed_text_parts=[],
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _fire_streamed_codex_commentary=lambda _text: None,
        _touch_activity=lambda _message: None,
        _client_log_context=lambda: "test",
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    request = {"model": "gpt-5", "input": "hello"}

    first = run_codex_stream(agent, request, client=client)
    second = run_codex_stream(agent, request, client=client)

    assert first.output == [output_item]
    assert second.output == [output_item]
    assert len(ws_calls) == 1
    assert len(sse_calls) == 2
    assert agent._generic_ws_auto_disabled_for is not None

    # Changing ws_url must re-enable WS attempts under auto mode.
    agent.responses_ws_url = "wss://relay.example.com/ws/responses"
    third = run_codex_stream(agent, request, client=client)
    assert third.output == [output_item]
    assert len(ws_calls) == 2
    assert len(sse_calls) == 3


def test_error_classifier_handles_generic_ws_errors():
    from agent.codex_responses_ws_transport import (
        GenericWsNotStartedError,
        GenericWsRejectedError,
        GenericWsStartedError,
    )
    from agent.error_classifier import FailoverReason, classify_api_error

    not_started = classify_api_error(GenericWsNotStartedError("upgrade failed"))
    assert not_started.reason == FailoverReason.timeout
    assert not_started.retryable is True
    assert not_started.should_fallback is True

    started = classify_api_error(GenericWsStartedError("after send"))
    assert started.retryable is False
    assert started.should_fallback is True

    rejected = classify_api_error(
        GenericWsRejectedError("bad request", status_code=400)
    )
    assert rejected.status_code == 400
    assert rejected.should_fallback is True
