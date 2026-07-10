import json
from types import SimpleNamespace

import pytest

from agent.codex_websocket import (
    CODEX_RESPONSES_LITE_CLIENT_METADATA_KEY,
    CODEX_RESPONSES_LITE_HEADER,
    CODEX_RESPONSES_WEBSOCKET_BETA,
    CodexWebSocketError,
    build_codex_websocket_headers,
    build_codex_websocket_request,
    responses_websocket_url,
    run_codex_websocket,
)


def _agent():
    return SimpleNamespace(
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        _client_kwargs={
            "default_headers": {
                "User-Agent": "codex_cli_rs/0.144.1 (Hermes Agent)",
                "originator": "codex_cli_rs",
            }
        },
        _interrupt_requested=False,
    )


def test_responses_websocket_url_rewrites_http_scheme():
    assert (
        responses_websocket_url("https://chatgpt.com/backend-api/codex")
        == "wss://chatgpt.com/backend-api/codex/responses"
    )
    assert (
        responses_websocket_url("https://chatgpt.com/backend-api/codex/responses")
        == "wss://chatgpt.com/backend-api/codex/responses"
    )


def test_build_websocket_request_matches_response_create_shape():
    request = build_codex_websocket_request(
        {
            "model": "gpt-5.6-luna",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "name": "terminal"}],
            "store": False,
            "parallel_tool_calls": True,
            "stream": True,
            "extra_headers": {"session_id": "session-1"},
            "extra_body": {"client_metadata": {"source": "test"}},
            "timeout": 30,
        },
        use_responses_lite=True,
    )

    assert request["type"] == "response.create"
    assert request["stream"] is True
    assert request["model"] == "gpt-5.6-luna"
    assert request["reasoning"] == {"context": "all_turns"}
    assert request["parallel_tool_calls"] is False
    assert "instructions" not in request
    assert "tools" not in request
    assert request["input"][:2] == [
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
    assert request["client_metadata"] == {
        "source": "test",
        CODEX_RESPONSES_LITE_CLIENT_METADATA_KEY: "true",
    }
    assert "extra_headers" not in request
    assert "extra_body" not in request
    assert "timeout" not in request


def test_build_websocket_request_does_not_duplicate_prepared_lite_input():
    request = build_codex_websocket_request(
        {
            "model": "gpt-5.6-luna",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "name": "terminal"}],
        },
        use_responses_lite=True,
    )

    rebuilt = build_codex_websocket_request(request, use_responses_lite=True)

    assert rebuilt["input"] == request["input"]
    assert rebuilt["input"][0]["type"] == "additional_tools"
    assert rebuilt["input"][1]["role"] == "developer"
    assert rebuilt["input"][2] == {"role": "user", "content": "hello"}


def test_build_websocket_headers_include_codex_transport_markers(monkeypatch):
    monkeypatch.setattr("agent.codex_websocket._codex_client_version", lambda: "0.144.1")

    headers = build_codex_websocket_headers(
        _agent(),
        {"extra_headers": {"session_id": "session-1"}},
        use_responses_lite=True,
    )

    assert headers["Authorization"] == "Bearer codex-token"
    assert headers["originator"] == "codex_cli_rs"
    assert headers["version"] == "0.144.1"
    assert headers["OpenAI-Beta"] == CODEX_RESPONSES_WEBSOCKET_BETA
    assert headers[CODEX_RESPONSES_LITE_HEADER] == "true"
    assert headers["session_id"] == "session-1"


class _FakeWebSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def recv(self, timeout=None):
        if not self.frames:
            raise AssertionError("test server ran out of frames")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


def test_run_codex_websocket_reuses_responses_event_consumer(monkeypatch):
    ws = _FakeWebSocket(
        [
            json.dumps({"type": "response.created"}),
            json.dumps({"type": "response.output_text.delta", "delta": "hello"}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {"id": "resp-1", "status": "completed", "usage": None},
                }
            ),
        ]
    )
    monkeypatch.setattr("websockets.sync.client.connect", lambda *args, **kwargs: ws)
    agent = _agent()

    response = run_codex_websocket(
        agent,
        {
            "model": "gpt-5.6-luna",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": "hello"}],
            "store": False,
            "parallel_tool_calls": True,
        },
        use_responses_lite=True,
    )

    assert response.id == "resp-1"
    assert response.output_text == "hello"
    assert response.output[0].content[0].text == "hello"
    assert ws.closed is True
    sent = json.loads(ws.sent[0])
    assert sent["type"] == "response.create"
    assert sent["model"] == "gpt-5.6-luna"
    assert sent["reasoning"]["context"] == "all_turns"
    assert sent["parallel_tool_calls"] is False
    assert sent["client_metadata"][CODEX_RESPONSES_LITE_CLIENT_METADATA_KEY] == "true"


def test_run_codex_websocket_uses_terminal_output_when_done_event_is_missing(monkeypatch):
    ws = _FakeWebSocket(
        [
            json.dumps({"type": "response.created"}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp-terminal-output",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {"type": "output_text", "text": "terminal hello"}
                                ],
                            }
                        ],
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr("websockets.sync.client.connect", lambda *args, **kwargs: ws)

    response = run_codex_websocket(
        _agent(),
        {
            "model": "gpt-5.6-luna",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": "hello"}],
            "store": False,
        },
        use_responses_lite=True,
    )

    assert response.output[0].content[0].text == "terminal hello"


def test_run_codex_websocket_marks_pre_response_errors_safe_to_fallback(monkeypatch):
    ws = _FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "error",
                    "status": 403,
                    "error": {"message": "websocket policy rejected"},
                }
            )
        ]
    )
    monkeypatch.setattr("websockets.sync.client.connect", lambda *args, **kwargs: ws)

    with pytest.raises(CodexWebSocketError) as exc_info:
        run_codex_websocket(
            _agent(),
            {
                "model": "gpt-5.6-luna",
                "instructions": "Be concise.",
                "input": [{"role": "user", "content": "hello"}],
                "store": False,
            },
            use_responses_lite=True,
        )

    assert exc_info.value.started is False
    assert exc_info.value.status_code == 403
