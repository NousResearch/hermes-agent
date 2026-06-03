import json
from types import SimpleNamespace

import pytest

import agent.codex_runtime as codex_runtime


def test_codex_responses_websocket_url_uses_provider_base_url():
    assert (
        codex_runtime.codex_responses_websocket_url("https://api.openai.com/v1")
        == "wss://api.openai.com/v1/responses"
    )
    assert (
        codex_runtime.codex_responses_websocket_url("http://localhost:8080/v1/")
        == "ws://localhost:8080/v1/responses"
    )
    assert (
        codex_runtime.codex_responses_websocket_url("wss://relay.example.com/v1/responses")
        == "wss://relay.example.com/v1/responses"
    )


def test_provider_config_enables_websocket_only_for_codex_responses(monkeypatch):
    config = {
        "providers": {
            "vendor": {
                "base_url": "https://api.vendor.example.com/v1",
                "transport": "codex_responses",
                "codex_responses_websocket": True,
            }
        }
    }
    monkeypatch.setattr(codex_runtime, "load_config", lambda: config, raising=False)

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="vendor",
        base_url="https://api.vendor.example.com/v1",
    )
    assert codex_runtime.codex_responses_websocket_enabled(agent) is True

    agent.api_mode = "chat_completions"
    assert codex_runtime.codex_responses_websocket_enabled(agent) is False


def test_provider_config_transport_string_can_enable_websocket(monkeypatch):
    config = {
        "providers": {
            "vendor": {
                "base_url": "https://api.vendor.example.com/v1",
                "transport": "codex_responses",
                "codex_responses_transport": "websocket",
            }
        }
    }
    monkeypatch.setattr(codex_runtime, "load_config", lambda: config, raising=False)

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="vendor",
        base_url="https://api.vendor.example.com/v1/",
    )
    assert codex_runtime.codex_responses_websocket_enabled(agent) is True


def test_codex_websocket_headers_append_responses_websockets_beta():
    agent = SimpleNamespace(api_key="sk-test")

    headers = codex_runtime._codex_websocket_headers(
        agent,
        {"extra_headers": {"OpenAI-Beta": "assistants=v2"}},
    )

    beta_values = {value.strip() for value in headers["OpenAI-Beta"].split(",")}
    assert beta_values == {"assistants=v2", "responses_websockets=2026-02-06"}


def test_codex_websocket_headers_include_codex_session_affinity_headers():
    agent = SimpleNamespace(
        api_key="sk-test",
        session_id="hermes-session-123",
        _thread_id="hermes-thread-456",
    )

    headers = codex_runtime._codex_websocket_headers(agent, {})

    assert headers["session-id"] == "hermes-session-123"
    assert headers["thread-id"] == "hermes-thread-456"
    assert headers["x-client-request-id"] == "hermes-thread-456"


def test_codex_websocket_headers_fall_back_to_session_id_for_thread_id():
    agent = SimpleNamespace(
        api_key="sk-test",
        session_id="hermes-session-123",
        _thread_id="",
    )

    headers = codex_runtime._codex_websocket_headers(agent, {})

    assert headers["session-id"] == "hermes-session-123"
    assert headers["thread-id"] == "hermes-session-123"
    assert headers["x-client-request-id"] == "hermes-session-123"


def test_codex_websocket_headers_fall_back_to_session_id_when_thread_id_missing():
    agent = SimpleNamespace(
        api_key="sk-test",
        session_id="hermes-session-123",
    )

    headers = codex_runtime._codex_websocket_headers(agent, {})

    assert headers["session-id"] == "hermes-session-123"
    assert headers["thread-id"] == "hermes-session-123"
    assert headers["x-client-request-id"] == "hermes-session-123"


def test_codex_websocket_headers_preserve_explicit_session_affinity_headers():
    agent = SimpleNamespace(
        api_key="sk-test",
        session_id="derived-session",
        _thread_id="derived-thread",
    )

    headers = codex_runtime._codex_websocket_headers(
        agent,
        {
            "extra_headers": {
                "session-id": "explicit-session",
                "thread-id": "explicit-thread",
                "x-client-request-id": "explicit-request",
            }
        },
    )

    assert headers["session-id"] == "explicit-session"
    assert headers["thread-id"] == "explicit-thread"
    assert headers["x-client-request-id"] == "explicit-request"


def test_codex_websocket_headers_preserve_explicit_session_headers_case_insensitively():
    agent = SimpleNamespace(
        api_key="sk-test",
        session_id="derived-session",
        _thread_id="derived-thread",
    )

    headers = codex_runtime._codex_websocket_headers(
        agent,
        {
            "extra_headers": {
                "Session-ID": "explicit-session",
                "Thread-ID": "explicit-thread",
                "X-Client-Request-ID": "explicit-request",
            }
        },
    )

    assert headers["Session-ID"] == "explicit-session"
    assert headers["Thread-ID"] == "explicit-thread"
    assert headers["X-Client-Request-ID"] == "explicit-request"
    assert "session-id" not in headers
    assert "thread-id" not in headers
    assert "x-client-request-id" not in headers


def test_provider_config_does_not_stop_at_same_url_without_switch(monkeypatch):
    config = {
        "providers": {
            "plain": {
                "base_url": "https://api.vendor.example.com/v1",
                "transport": "codex_responses",
            },
            "vendor": {
                "base_url": "https://api.vendor.example.com/v1",
                "transport": "codex_responses",
                "codex_responses_websocket": True,
            },
        }
    }
    monkeypatch.setattr(codex_runtime, "load_config", lambda: config, raising=False)

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="vendor",
        base_url="https://api.vendor.example.com/v1/",
    )
    assert codex_runtime.codex_responses_websocket_enabled(agent) is True


def test_provider_name_match_does_not_inherit_same_url_switch(monkeypatch):
    config = {
        "providers": {
            "plain": {
                "base_url": "https://api.vendor.example.com/v1",
                "transport": "codex_responses",
            },
            "vendor": {
                "base_url": "https://api.vendor.example.com/v1",
                "transport": "codex_responses",
                "codex_responses_websocket": True,
            },
        }
    }
    monkeypatch.setattr(codex_runtime, "load_config", lambda: config, raising=False)

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="plain",
        base_url="https://api.vendor.example.com/v1/",
    )
    assert codex_runtime.codex_responses_websocket_enabled(agent) is False


def test_provider_config_can_fallback_to_base_url_for_custom_provider(monkeypatch):
    config = {
        "custom_providers": [
            {
                "name": "vendor",
                "base_url": "https://api.vendor.example.com/v1",
                "api_mode": "codex_responses",
                "codex_responses_websocket": True,
            }
        ]
    }
    monkeypatch.setattr(codex_runtime, "load_config", lambda: config, raising=False)

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="custom",
        base_url="https://api.vendor.example.com/v1/",
    )
    assert codex_runtime.codex_responses_websocket_enabled(agent) is True


class _FakeWebSocket:
    def __init__(self, events):
        self.events = list(events)
        self.sent_payloads = []
        self.recv_timeouts = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    def send(self, payload):
        self.sent_payloads.append(json.loads(payload))

    def recv(self, timeout=None):
        self.recv_timeouts.append(timeout)
        if not self.events:
            raise AssertionError("fake websocket received too many recv calls")
        return json.dumps(self.events.pop(0))


def test_run_codex_stream_uses_provider_websocket_payload(monkeypatch):
    events = [
        {"type": "response.output_text.delta", "delta": "hello"},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4,
                },
            },
        },
    ]
    fake_ws = _FakeWebSocket(events)
    connect_calls = []

    def fake_connect(uri, **kwargs):
        connect_calls.append({"uri": uri, **kwargs})
        return fake_ws

    monkeypatch.setattr(codex_runtime, "codex_responses_websocket_enabled", lambda agent: True)
    monkeypatch.setattr(codex_runtime, "_connect_responses_websocket", fake_connect, raising=False)

    deltas = []
    touches = []
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="vendor",
        base_url="https://api.vendor.example.com/v1",
        api_key="sk-test",
        _interrupt_requested=False,
        _fire_stream_delta=deltas.append,
        _fire_reasoning_delta=lambda text: None,
        _touch_activity=touches.append,
        _client_log_context=lambda: "test-context",
    )
    api_kwargs = {
        "model": "gpt-5-codex",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": "Ping"}],
        "store": False,
        "timeout": 30,
        "stream": True,
        "extra_headers": {"X-Provider-Header": "abc"},
        "extra_body": {"metadata": {"tenant": "vendor"}},
    }

    response = codex_runtime.run_codex_stream(agent, api_kwargs)

    assert connect_calls[0]["uri"] == "wss://api.vendor.example.com/v1/responses"
    assert connect_calls[0]["additional_headers"]["Authorization"] == "Bearer sk-test"
    assert connect_calls[0]["additional_headers"]["X-Provider-Header"] == "abc"
    assert connect_calls[0]["additional_headers"]["OpenAI-Beta"] == "responses_websockets=2026-02-06"

    sent = fake_ws.sent_payloads[0]
    assert sent["type"] == "response.create"
    assert sent["model"] == "gpt-5-codex"
    assert sent["metadata"] == {"tenant": "vendor"}
    assert "extra_body" not in sent
    assert "extra_headers" not in sent
    assert "timeout" not in sent
    assert "stream" not in sent

    assert deltas == ["hello"]
    assert response.id == "resp_1"
    assert response.output[0].content[0].text == "hello"
    assert response.usage.input_tokens == 3
    assert fake_ws.closed is False
    codex_runtime.close_codex_responses_websocket_session(agent)
    assert fake_ws.closed is True


def test_websocket_stream_json_events_normalize_function_call_items(monkeypatch):
    fake_ws = _FakeWebSocket([
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "terminal",
                "arguments": "{\"cmd\":\"pwd\"}",
            },
        },
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed"},
        },
    ])

    monkeypatch.setattr(codex_runtime, "codex_responses_websocket_enabled", lambda agent: True)
    monkeypatch.setattr(
        codex_runtime,
        "_connect_responses_websocket",
        lambda uri, **kwargs: fake_ws,
        raising=False,
    )

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="vendor",
        base_url="https://api.vendor.example.com/v1",
        api_key="sk-test",
        _interrupt_requested=False,
        _fire_stream_delta=lambda text: None,
        _fire_reasoning_delta=lambda text: None,
        _touch_activity=lambda reason: None,
        _client_log_context=lambda: "test-context",
    )
    response = codex_runtime.run_codex_stream(
        agent,
        {
            "model": "gpt-5-codex",
            "instructions": "You are Hermes.",
            "input": [{"role": "user", "content": "Ping"}],
            "store": False,
        },
    )

    assert response.output[0].type == "function_call"
    assert response.output[0].name == "terminal"


def test_websocket_session_builds_incremental_previous_response_payload():
    session = codex_runtime._CodexResponsesWebSocketSession(
        uri="wss://api.vendor.example.com/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        open_timeout=10,
    )
    session.previous_response_id = "resp_1"
    session.last_full_input = [
        {"role": "user", "content": "Run pwd"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "terminal",
            "arguments": "{\"cmd\":\"pwd\"}",
        },
    ]

    payload, full_input, used_continuation = session.build_payload(
        {
            "model": "gpt-5-codex",
            "input": [
                {"role": "user", "content": "Run pwd"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "terminal",
                    "arguments": "{\"cmd\":\"pwd\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"cwd\":\"/workspace\"}",
                },
            ],
            "store": False,
            "stream": True,
        }
    )

    assert used_continuation is True
    assert payload["type"] == "response.create"
    assert payload["previous_response_id"] == "resp_1"
    assert payload["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "{\"cwd\":\"/workspace\"}",
        }
    ]
    assert full_input[-1]["type"] == "function_call_output"


def test_websocket_session_falls_back_to_full_input_when_history_is_not_prefix():
    session = codex_runtime._CodexResponsesWebSocketSession(
        uri="wss://api.vendor.example.com/v1/responses",
        headers={"Authorization": "Bearer sk-test"},
        open_timeout=10,
    )
    session.previous_response_id = "resp_1"
    session.last_full_input = [{"role": "user", "content": "old"}]

    payload, _full_input, used_continuation = session.build_payload(
        {
            "model": "gpt-5-codex",
            "input": [{"role": "user", "content": "fresh"}],
            "store": False,
        }
    )

    assert used_continuation is False
    assert "previous_response_id" not in payload
    assert payload["input"] == [{"role": "user", "content": "fresh"}]


def test_websocket_transport_close_with_code_can_fallback_to_http():
    class ConnectionClosedError(Exception):
        code = 1006
        reason = "abnormal closure"

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="vendor",
        base_url="https://api.vendor.example.com/v1",
        _codex_responses_websocket_output_committed=False,
    )

    assert codex_runtime.should_fallback_codex_responses_websocket_to_http(
        agent,
        ConnectionClosedError("websocket connection closed"),
    ) is True
