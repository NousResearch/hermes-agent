"""End-to-end recovery tests for OpenAI-compatible output-cap overflows."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


CONTEXT_LENGTH = 131_072
DEFAULT_OUTPUT_CAP = 65_536


def _content_chars(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_content_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(_content_chars(item) for item in value.values())
    return 0


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    tool_responses_remaining = 0

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return

        type(self).requests.append(request)
        input_tokens = max(
            1,
            sum(_content_chars(message.get("content")) for message in request["messages"])
            // 4,
        )
        output_cap = request.get("max_tokens", DEFAULT_OUTPUT_CAP)
        if input_tokens + output_cap > CONTEXT_LENGTH:
            # This reproduces vLLM's derived lower bound exactly. The reported
            # input is computed from the rejected cap, not measured.
            reported_input = CONTEXT_LENGTH + 1 - output_cap
            message = (
                "This model's maximum context length is 131072 tokens. However, "
                f"you requested {output_cap} output tokens and your prompt contains "
                f"at least {reported_input} input tokens, for a total of at least "
                "131073 tokens. Please reduce the length of the input prompt or "
                "the number of requested output tokens. (parameter=input_tokens, "
                f"value={reported_input})"
            )
            self._send_json(400, {"error": {"message": message}})
            return

        tool_call = type(self).tool_responses_remaining > 0
        if tool_call:
            type(self).tool_responses_remaining -= 1

        if request.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in self._stream_chunks(tool_call):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        message: dict = {"role": "assistant", "content": "Recovered"}
        finish_reason = "stop"
        if tool_call:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [self._tool_call()],
            }
            finish_reason = "tool_calls"
        self._send_json(
            200,
            {
                "id": "output-cap-recovery",
                "object": "chat.completion",
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": 1,
                    "total_tokens": input_tokens + 1,
                },
            },
        )

    @classmethod
    def _tool_call(cls) -> dict:
        index = len(cls.requests)
        return {
            "id": f"call-large-result-{index}",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": f'{{"query":"large-{index}"}}',
            },
        }

    @classmethod
    def _stream_chunks(cls, tool_call: bool) -> list[dict]:
        deltas = [{"role": "assistant", "content": ""}]
        if tool_call:
            call = cls._tool_call()
            call["index"] = 0
            deltas.append({"tool_calls": [call]})
            finish_reason = "tool_calls"
        else:
            deltas.append({"content": "Recovered"})
            finish_reason = "stop"
        return [
            {
                "id": "output-cap-recovery",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None}
                ],
            }
            for delta in deltas
        ] + [
            {
                "id": "output-cap-recovery",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish_reason}
                ],
            }
        ]

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def openai_compatible_server():
    _OpenAICompatibleHandler.requests = []
    _OpenAICompatibleHandler.tool_responses_remaining = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _make_agent(tmp_path, monkeypatch, base_url, *, max_iterations=3):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="custom",
        model="local-model",
        max_iterations=max_iterations,
        disabled_toolsets=["*"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        save_trajectories=False,
        session_db=None,
    )
    agent.context_compressor.context_length = CONTEXT_LENGTH
    assert agent.max_tokens is None
    return agent


@pytest.mark.parametrize("streaming", [True, False])
def test_output_cap_overflow_recovers_through_real_turn_path(
    tmp_path, monkeypatch, openai_compatible_server, streaming, caplog
):
    """A near-full custom-provider session clamps and completes the turn."""
    agent = _make_agent(tmp_path, monkeypatch, openai_compatible_server)
    agent.context_compressor.threshold_tokens = 98_304
    agent._disable_streaming = not streaming
    monkeypatch.setattr(
        agent,
        "_compress_context",
        lambda messages, system_message, **_kwargs: (messages, system_message),
    )
    caplog.set_level(logging.DEBUG, logger="agent.conversation_loop")

    history = [
        {"role": "user", "content": "x" * 262_144},
        {"role": "assistant", "content": "ack"},
    ]
    result = agent.run_conversation(
        "continue",
        conversation_history=history,
        stream_callback=(lambda _delta: None) if streaming else None,
    )

    requests = _OpenAICompatibleHandler.requests
    assert result["completed"] is True
    assert result["final_response"] == "Recovered"
    assert result.get("compression_exhausted") is not True
    assert len(requests) == 2
    assert (requests[0].get("stream") is True) is streaming
    assert requests[0]["max_tokens"] == DEFAULT_OUTPUT_CAP
    assert requests[1]["max_tokens"] < requests[0]["max_tokens"] // 2 + 1
    assert (requests[1].get("stream") is True) is streaming
    assert any(
        "API call failed full error" in record.getMessage()
        and "value=65537" in record.getMessage()
        for record in caplog.records
    )


def test_output_cap_retry_is_independent_of_compression_budget(
    tmp_path, monkeypatch, openai_compatible_server
):
    """Prior compactions cannot cancel the first reduced-cap request."""
    _OpenAICompatibleHandler.tool_responses_remaining = 3
    agent = _make_agent(
        tmp_path, monkeypatch, openai_compatible_server, max_iterations=4
    )
    agent.context_compressor.threshold_tokens = 20_000
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Return a test payload",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent.valid_tool_names = {"web_search"}

    monkeypatch.setattr(
        agent,
        "_compress_context",
        lambda messages, system_message, **_kwargs: (messages, system_message),
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda *_args, **_kwargs: "x" * 90_000,
    )
    monkeypatch.setattr(
        "agent.tool_executor.maybe_persist_tool_result",
        lambda content, *_args, **_kwargs: content,
    )
    monkeypatch.setattr(
        "agent.tool_executor.enforce_turn_budget",
        lambda *_args, **_kwargs: None,
    )

    result = agent.run_conversation("fetch several large results")

    requests = _OpenAICompatibleHandler.requests
    assert result["completed"] is True
    assert result["final_response"] == "Recovered"
    assert result.get("compression_exhausted") is not True
    assert [request["max_tokens"] for request in requests] == [
        DEFAULT_OUTPUT_CAP,
        DEFAULT_OUTPUT_CAP,
        DEFAULT_OUTPUT_CAP,
        DEFAULT_OUTPUT_CAP,
        32_704,
    ]


def test_output_cap_retry_preserves_later_compression_budget(
    tmp_path, monkeypatch, openai_compatible_server
):
    """An output-cap retry cannot spend a later compression attempt."""
    _OpenAICompatibleHandler.tool_responses_remaining = 1
    agent = _make_agent(
        tmp_path, monkeypatch, openai_compatible_server, max_iterations=2
    )
    agent.max_compression_attempts = 1
    agent.context_compressor.threshold_tokens = 98_304
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Return a test payload",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent.valid_tool_names = {"web_search"}

    compression_request_counts = []
    tool_response_request_counts = []

    def compress(messages, system_message, **_kwargs):
        compression_request_counts.append(len(_OpenAICompatibleHandler.requests))
        return messages, system_message

    def large_tool_result(*_args, **_kwargs):
        tool_response_request_counts.append(len(_OpenAICompatibleHandler.requests))
        agent.context_compressor.last_prompt_tokens = 0
        return "x" * 200_000

    monkeypatch.setattr(agent, "_compress_context", compress)
    monkeypatch.setattr("run_agent.handle_function_call", large_tool_result)
    monkeypatch.setattr(
        "agent.tool_executor.maybe_persist_tool_result",
        lambda content, *_args, **_kwargs: content,
    )
    monkeypatch.setattr(
        "agent.tool_executor.enforce_turn_budget",
        lambda *_args, **_kwargs: None,
    )

    history = [
        {"role": "user", "content": "x" * 262_144},
        {"role": "assistant", "content": "ack"},
    ]
    result = agent.run_conversation("continue", conversation_history=history)

    assert result["completed"] is True
    assert len(tool_response_request_counts) == 1
    assert tool_response_request_counts[0] in compression_request_counts


def test_successful_provider_response_rearms_output_cap_budget(
    tmp_path, monkeypatch, openai_compatible_server
):
    """Later tool iterations receive a fresh bounded clamp opportunity."""
    _OpenAICompatibleHandler.tool_responses_remaining = 1
    agent = _make_agent(
        tmp_path, monkeypatch, openai_compatible_server, max_iterations=2
    )
    agent.max_compression_attempts = 1
    agent.context_compressor.threshold_tokens = 98_304
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Return a test payload",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    agent.valid_tool_names = {"web_search"}
    monkeypatch.setattr(
        agent,
        "_compress_context",
        lambda messages, system_message, **_kwargs: (messages, system_message),
    )
    monkeypatch.setattr("run_agent.handle_function_call", lambda *_a, **_k: "ok")

    history = [
        {"role": "user", "content": "x" * 262_144},
        {"role": "assistant", "content": "ack"},
    ]
    result = agent.run_conversation("continue", conversation_history=history)

    assert result["completed"] is True
    caps = [request["max_tokens"] for request in _OpenAICompatibleHandler.requests]
    assert caps == [DEFAULT_OUTPUT_CAP, 32_704, DEFAULT_OUTPUT_CAP, 32_704]
