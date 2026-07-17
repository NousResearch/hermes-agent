import json
import threading
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


class _BlockingFakeWebSocket(_FakeWebSocket):
    """Socket that stays in recv until a cross-thread close wakes it."""

    def __init__(self):
        super().__init__([])
        self.recv_started = threading.Event()
        self.closed_event = threading.Event()
        self.recv_thread_id = None
        self.close_thread_ids = []

    def recv(self, timeout=None):
        self.recv_thread_id = threading.get_ident()
        self.recv_started.set()
        if not self.closed_event.wait(timeout=timeout):
            raise TimeoutError
        return None

    def close(self):
        self.close_thread_ids.append(threading.get_ident())
        self.closed = True
        self.closed_event.set()


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


def test_abort_hook_closes_active_websocket_without_http_replay(monkeypatch):
    """A cross-thread abort must close, clear, and never replay a sent request."""
    from agent.codex_runtime import run_codex_stream
    from run_agent import AIAgent

    ws = _BlockingFakeWebSocket()
    monkeypatch.setattr("websockets.sync.client.connect", lambda *args, **kwargs: ws)
    monkeypatch.setattr(
        "agent.codex_runtime._codex_model_capabilities",
        lambda _agent: SimpleNamespace(
            use_responses_lite=True,
            prefer_websockets=True,
            should_use_websocket=True,
        ),
    )

    agent = _agent()
    agent.provider = "openai-codex"
    agent.model = "gpt-5.6-luna"
    agent._base_url_lower = agent.base_url
    agent._base_url_hostname = "chatgpt.com"
    agent._fire_stream_delta = lambda text: None
    agent._fire_reasoning_delta = lambda text: None
    agent._touch_activity = lambda message: None
    agent._client_log_context = lambda: "test-context"
    agent._force_close_tcp_sockets = lambda client: 0

    http_calls = []

    def create(**kwargs):
        http_calls.append(kwargs)
        raise AssertionError("a sent WebSocket request must not be replayed over HTTP")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    outcome = {}

    def run_request():
        try:
            outcome["response"] = run_codex_stream(
                agent,
                {
                    "model": "gpt-5.6-luna",
                    "instructions": "Be concise.",
                    "input": [{"role": "user", "content": "hello"}],
                    "store": False,
                },
                client=client,
            )
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run_request, daemon=True)
    worker.start()
    recv_started = ws.recv_started.wait(timeout=3.0)
    if not recv_started:
        ws.close()
        worker.join(timeout=3.0)
    assert recv_started, f"WebSocket request did not reach recv: {outcome!r}"

    abort_thread_id = threading.get_ident()
    close_count_before_abort = len(ws.close_thread_ids)
    AIAgent._abort_request_openai_client(agent, client, reason="interrupt_abort")
    abort_close_thread_ids = ws.close_thread_ids[close_count_before_abort:]
    worker.join(timeout=3.0)
    stopped_by_abort = not worker.is_alive()
    if not stopped_by_abort:
        ws.close()
        worker.join(timeout=3.0)

    assert stopped_by_abort
    assert len(ws.sent) == 1
    assert ws.closed is True
    assert abort_thread_id in abort_close_thread_ids
    assert ws.recv_thread_id != abort_thread_id
    assert getattr(agent, "_codex_active_websocket", None) is None
    assert http_calls == []
    assert isinstance(outcome.get("error"), CodexWebSocketError)
    assert outcome["error"].safe_to_fallback is False
