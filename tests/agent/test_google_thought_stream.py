"""Vertex / Google OpenAI-compat streaming thought chunks route to reasoning, not content."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from openai import OpenAI

from agent.auxiliary_client import _ChatStreamAccumulator
from agent.chat_completion_helpers import build_assistant_message, interruptible_streaming_api_call
from agent.chat_completion_helpers_relay import RelayChatAccumulator
from agent.reasoning_carriers import is_google_thought_delta
from agent.transports.chat_completions import ChatCompletionsTransport
from run_agent import AIAgent


def test_is_google_thought_delta_shapes():
    assert is_google_thought_delta({"extra_content": {"google": {"thought": True}}}) is True
    assert is_google_thought_delta({"extra_content": {"thought": True}}) is True
    sdk_like = SimpleNamespace(content="x", model_extra={"extra_content": {"google": {"thought": True}}})
    assert is_google_thought_delta(sdk_like) is True

    assert is_google_thought_delta({"extra_content": {"google": {"thought": False}}}) is False
    assert is_google_thought_delta({"extra_content": {"google": {"thought_signature": "SIG"}}}) is False
    assert is_google_thought_delta({"content": "hello"}) is False
    assert is_google_thought_delta(SimpleNamespace(content="hello")) is False


def _serve_sse(chunks: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.parametrize("collector", ["main", "relay", "aux"])
def test_vertex_google_thought_chunks_route_to_reasoning_only(collector):
    deltas = [
        {"role": "assistant", "content": "**Estimating Progress**\n\nStep 1.", "extra_content": {"google": {"thought": True}}},
        {"role": "assistant", "content": "**Checking Result**\n\nStep 2.", "extra_content": {"google": {"thought": True}}},
        {"role": "assistant", "content": "Final "},
        {"role": "assistant", "content": "answer."},
        {"role": "assistant", "content": "", "extra_content": {"google": {"thought_signature": "SIG_TEXT"}}},
    ]
    chunks = [
        {
            "id": "vertex-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "google/gemini-4-neon",
            "choices": [{"index": 0, "delta": d, "finish_reason": "stop" if i == len(deltas) - 1 else None}],
        }
        for i, d in enumerate(deltas)
    ]
    server, thread = _serve_sse(chunks)
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    kwargs = {"model": "google/gemini-4-neon", "messages": [{"role": "user", "content": "hi"}]}
    try:
        if collector == "main":
            streamed_text: list[str] = []
            streamed_reasoning: list[str] = []
            agent = AIAgent(
                api_key="local-test",
                base_url=base_url,
                provider="vertex",
                model="google/gemini-4-neon",
                platform="subagent",
                quiet_mode=True,
                skip_memory=True,
                skip_context_files=True,
                enabled_toolsets=[],
                save_trajectories=False,
                stream_delta_callback=streamed_text.append,
                reasoning_callback=streamed_reasoning.append,
            )
            response = interruptible_streaming_api_call(agent, kwargs)
            assert response is not None
            assert "".join(streamed_text) == "Final answer."
            assert "".join(streamed_reasoning) == "**Estimating Progress**\n\nStep 1.\n\n**Checking Result**\n\nStep 2."
            msg = response.choices[0].message
            assert msg.content == "Final answer."
            assert msg.reasoning_content == "**Estimating Progress**\n\nStep 1.\n\n**Checking Result**\n\nStep 2."
            assert msg.extra_content == {"google": {"thought_signature": "SIG_TEXT"}}
            normalized = ChatCompletionsTransport().normalize_response(response)
            stored = build_assistant_message(agent, normalized, normalized.finish_reason)
            assert stored["content"] == "Final answer."
            assert stored["reasoning"] == "**Estimating Progress**\n\nStep 1.\n\n**Checking Result**\n\nStep 2."
        elif collector == "relay":
            acc = RelayChatAccumulator()
            with OpenAI(api_key="local-test", base_url=base_url) as client:
                with client.chat.completions.create(**kwargs, stream=True) as stream:
                    for chunk in stream:
                        acc.observe(chunk.model_dump(mode="json"))
            msg = acc.finalize()["choices"][0]["message"]
            assert msg["content"] == "Final answer."
            assert msg["reasoning_content"] == "**Estimating Progress**\n\nStep 1.\n\n**Checking Result**\n\nStep 2."
        else:
            acc = _ChatStreamAccumulator(model="google/gemini-4-neon")
            with OpenAI(api_key="local-test", base_url=base_url) as client:
                with client.chat.completions.create(**kwargs, stream=True) as stream:
                    for chunk in stream:
                        acc.feed(chunk)
            msg = acc.finish().choices[0].message
            assert msg.content == "Final answer."
            assert msg.reasoning == "**Estimating Progress**\n\nStep 1.**Checking Result**\n\nStep 2."
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_vertex_thought_before_tool_call_leaves_content_empty():
    deltas = [
        {"role": "assistant", "content": "**Locating Recording**\n\nSearching.", "extra_content": {"google": {"thought": True}}},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_meetings", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "SIG_TC"}},
                }
            ],
        },
        {"role": "assistant", "content": ""},
    ]
    chunks = [
        {
            "id": "vertex-2",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "google/gemini-4-neon",
            "choices": [{"index": 0, "delta": d, "finish_reason": "tool_calls" if i == len(deltas) - 1 else None}],
        }
        for i, d in enumerate(deltas)
    ]
    server, thread = _serve_sse(chunks)
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        streamed_text: list[str] = []
        streamed_reasoning: list[str] = []
        agent = AIAgent(
            api_key="local-test",
            base_url=base_url,
            provider="vertex",
            model="google/gemini-4-neon",
            platform="subagent",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            enabled_toolsets=[],
            save_trajectories=False,
            stream_delta_callback=streamed_text.append,
            reasoning_callback=streamed_reasoning.append,
        )
        response = interruptible_streaming_api_call(
            agent, {"model": "google/gemini-4-neon", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response is not None
        assert streamed_text == []
        assert "".join(streamed_reasoning) == "**Locating Recording**\n\nSearching."
        msg = response.choices[0].message
        assert msg.content is None
        assert msg.reasoning_content == "**Locating Recording**\n\nSearching."
        assert len(msg.tool_calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_vertex_thought_and_tool_call_in_same_delta_preserves_both():
    deltas = [
        {
            "role": "assistant",
            "content": "**Calling Tool**\n\nFetching weather.",
            "extra_content": {"google": {"thought": True}},
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_combo",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    "extra_content": {"google": {"thought_signature": "SIG_COMBO"}},
                }
            ],
        },
        {"role": "assistant", "content": ""},
    ]
    chunks = [
        {
            "id": "vertex-3",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "google/gemini-4-neon",
            "choices": [{"index": 0, "delta": d, "finish_reason": "tool_calls" if i == len(deltas) - 1 else None}],
        }
        for i, d in enumerate(deltas)
    ]
    server, thread = _serve_sse(chunks)
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        streamed_text: list[str] = []
        streamed_reasoning: list[str] = []
        agent = AIAgent(
            api_key="local-test",
            base_url=base_url,
            provider="vertex",
            model="google/gemini-4-neon",
            platform="subagent",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            enabled_toolsets=[],
            save_trajectories=False,
            stream_delta_callback=streamed_text.append,
            reasoning_callback=streamed_reasoning.append,
        )
        response = interruptible_streaming_api_call(
            agent, {"model": "google/gemini-4-neon", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response is not None
        assert streamed_text == []
        assert "".join(streamed_reasoning) == "**Calling Tool**\n\nFetching weather."
        msg = response.choices[0].message
        assert msg.content is None
        assert msg.reasoning_content == "**Calling Tool**\n\nFetching weather."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
