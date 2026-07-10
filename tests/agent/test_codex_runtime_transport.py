from types import SimpleNamespace

import pytest

from agent.codex_websocket import CodexWebSocketError
from agent.codex_runtime import run_codex_stream


class _Stream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def _agent():
    return SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-luna",
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        _base_url_lower="https://chatgpt.com/backend-api/codex",
        _base_url_hostname="chatgpt.com",
        _interrupt_requested=False,
        _codex_streamed_text_parts=[],
        _fire_stream_delta=lambda text: None,
        _fire_reasoning_delta=lambda text: None,
        _touch_activity=lambda message: None,
        _client_log_context=lambda: "test-context",
    )


def _request():
    return {
        "model": "gpt-5.6-luna",
        "instructions": "Be concise.",
        "input": [{"role": "user", "content": "hello"}],
        "store": False,
    }


def _completed_stream():
    return _Stream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="hello"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp-1", status="completed", usage=None),
            ),
        ]
    )


def test_lite_capability_is_added_to_http_fallback_request(monkeypatch):
    agent = _agent()
    captured = {}
    monkeypatch.setattr(
        "agent.codex_runtime._codex_model_capabilities",
        lambda _agent: SimpleNamespace(
            use_responses_lite=True,
            prefer_websockets=False,
            should_use_websocket=False,
        ),
    )

    def create(**kwargs):
        captured.update(kwargs)
        return _completed_stream()

    request = _request()
    request["tools"] = [{"type": "function", "name": "terminal"}]
    request["input"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc",
                    "detail": "high",
                },
                {
                    "type": "input_image",
                    "image_url": "https://example.com/image.png",
                    "detail": "high",
                },
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,def",
                    "detail": "low",
                },
            ],
        }
    ]
    response = run_codex_stream(
        agent,
        request,
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )

    assert response.output_text == "hello"
    assert captured["extra_headers"]["X-OpenAI-Internal-Codex-Responses-Lite"] == "true"
    assert captured["reasoning"]["context"] == "all_turns"
    assert captured["stream"] is True
    assert "instructions" not in captured
    assert "tools" not in captured
    assert captured["input"][:2] == [
        {
            "type": "additional_tools",
            "role": "developer",
            "tools": [{"type": "function", "name": "terminal"}],
        },
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Be concise."}],
        },
    ]
    assert captured["input"][2]["content"] == [
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc",
        },
        {
            "type": "input_text",
            "text": "image content omitted because remote image URLs are not supported",
        },
        {
            "type": "input_text",
            "text": (
                "image content omitted because detail 'low' is not supported; "
                "use 'high', 'original', or 'auto'"
            ),
        },
    ]
    # Preparing the wire payload must not mutate the reusable conversation
    # request held by the agent loop.
    assert request["instructions"] == "Be concise."
    assert request["tools"] == [{"type": "function", "name": "terminal"}]
    assert request["input"][0]["content"][0]["detail"] == "high"
    assert request["input"][0]["content"][1]["image_url"].startswith("https://")
    assert request["input"][0]["content"][2]["detail"] == "low"


def test_lite_disables_parallel_tool_calls(monkeypatch):
    agent = _agent()
    captured = {}
    monkeypatch.setattr(
        "agent.codex_runtime._codex_model_capabilities",
        lambda _agent: SimpleNamespace(
            use_responses_lite=True,
            prefer_websockets=False,
            should_use_websocket=False,
        ),
    )

    def create(**kwargs):
        captured.update(kwargs)
        return _completed_stream()

    request = _request()
    request["parallel_tool_calls"] = True
    response = run_codex_stream(
        agent,
        request,
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )

    assert response.output_text == "hello"
    assert captured["parallel_tool_calls"] is False


def test_websocket_failure_before_events_falls_back_to_http(monkeypatch):
    agent = _agent()
    calls = {"websocket": 0, "http": 0}
    monkeypatch.setattr(
        "agent.codex_runtime._codex_model_capabilities",
        lambda _agent: SimpleNamespace(
            use_responses_lite=True,
            prefer_websockets=True,
            should_use_websocket=True,
        ),
    )

    def websocket(*args, **kwargs):
        calls["websocket"] += 1
        raise CodexWebSocketError("policy rejected", started=False, status_code=403)

    monkeypatch.setattr("agent.codex_websocket.run_codex_websocket", websocket)

    def create(**kwargs):
        calls["http"] += 1
        assert kwargs["extra_headers"]["X-OpenAI-Internal-Codex-Responses-Lite"] == "true"
        return _completed_stream()

    response = run_codex_stream(
        agent,
        _request(),
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )

    assert response.output_text == "hello"
    assert calls == {"websocket": 1, "http": 1}


def test_websocket_failure_after_events_is_not_replayed_over_http(monkeypatch):
    agent = _agent()
    calls = {"http": 0}
    monkeypatch.setattr(
        "agent.codex_runtime._codex_model_capabilities",
        lambda _agent: SimpleNamespace(
            use_responses_lite=True,
            prefer_websockets=True,
            should_use_websocket=True,
        ),
    )
    monkeypatch.setattr(
        "agent.codex_websocket.run_codex_websocket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CodexWebSocketError("stream dropped", started=True)
        ),
    )

    def create(**kwargs):
        calls["http"] += 1
        return _completed_stream()

    with pytest.raises(CodexWebSocketError, match="stream dropped"):
        run_codex_stream(
            agent,
            _request(),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        )

    assert calls["http"] == 0


def test_websocket_failure_after_request_send_is_not_replayed_over_http(monkeypatch):
    agent = _agent()
    calls = {"http": 0}
    monkeypatch.setattr(
        "agent.codex_runtime._codex_model_capabilities",
        lambda _agent: SimpleNamespace(
            use_responses_lite=True,
            prefer_websockets=True,
            should_use_websocket=True,
        ),
    )
    monkeypatch.setattr(
        "agent.codex_websocket.run_codex_websocket",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CodexWebSocketError(
                "socket dropped after request send",
                started=False,
                safe_to_fallback=False,
            )
        ),
    )

    def create(**kwargs):
        calls["http"] += 1
        return _completed_stream()

    with pytest.raises(CodexWebSocketError, match="socket dropped"):
        run_codex_stream(
            agent,
            _request(),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        )

    assert calls["http"] == 0
