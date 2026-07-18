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


class _FakeSocket:
    def __init__(self, frames=(), send_error=None):
        self._frames = iter(frames)
        self._send_error = send_error
        self.sent = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def send(self, payload):
        self.sent.append(payload)
        if self._send_error is not None:
            raise self._send_error

    def recv(self):
        return next(self._frames)

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
        return iter([
            SimpleNamespace(type="response.output_item.done", item=output_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed"),
            ),
        ])

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
