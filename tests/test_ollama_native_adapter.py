"""Tests for agent/ollama_native_adapter.py — the native Ollama /api/chat adapter.

Covers request/response/stream/embeddings translation, the flag-gated Ollama
detection probe, and the OpenAI-SDK-shaped client facade. Network is faked with
httpx.MockTransport + unittest.mock (no extra test dependency).
"""

import asyncio
import json
import os
from unittest.mock import patch

import httpx

from agent.ollama_native_adapter import (
    _OLLAMA_PROBE_CACHE,
    AsyncOllamaNativeClient,
    OllamaNativeClient,
    _translate_embeddings,
    build_ollama_request,
    is_native_ollama_base_url,
    native_root,
    translate_ollama_response,
    translate_stream_line,
)


def _mock_client(handler):
    """A real OllamaNativeClient wired to an httpx.MockTransport request handler."""
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── request construction ──────────────────────────────────────────────────────


def test_num_ctx_preserved_in_native_payload():
    req = build_ollama_request(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "hi"}],
        options={"num_ctx": 64000},
    )
    assert req["options"]["num_ctx"] == 64000
    assert req["stream"] is False


def test_sampling_merges_without_clobbering_options():
    req = build_ollama_request(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.5,
        top_p=0.9,
        max_tokens=128,
        seed=7,
        stop="STOP",
        options={"num_ctx": 8192, "temperature": 0.1},
    )
    opts = req["options"]
    assert opts["num_ctx"] == 8192
    assert opts["temperature"] == 0.1  # explicit option wins over the 0.5 arg
    assert opts["top_p"] == 0.9
    assert opts["num_predict"] == 128  # max_tokens -> num_predict
    assert opts["seed"] == 7
    assert opts["stop"] == ["STOP"]


def test_assistant_tool_call_args_string_to_object_and_tool_role():
    req = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "lookup", "arguments": '{"q": "cats"}'},
                    }
                ],
            },
            {"role": "tool", "name": "lookup", "content": "result"},
        ],
    )
    assert req["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"q": "cats"}
    assert req["messages"][1]["tool_name"] == "lookup"


def test_multimodal_image_data_url_extracted():
    req = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ],
    )
    assert req["messages"][0]["content"] == "what is this"
    assert req["messages"][0]["images"] == ["QUJD"]


# ── response + stream + embeddings translation ────────────────────────────────


def test_response_content_usage_and_finish_reason():
    resp = translate_ollama_response(
        {
            "model": "m",
            "message": {"role": "assistant", "content": "hello"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
        model="m",
    )
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == 15


def test_response_tool_calls_object_to_string():
    resp = translate_ollama_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            }
        },
        model="m",
    )
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.content is None
    assert msg.tool_calls[0].function.name == "f"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"a": 1}


def test_stream_content_tool_call_and_done():
    tool_idx = {"n": 0}
    c1 = translate_stream_line({"message": {"content": "hi"}}, "m", tool_idx)
    assert c1[0].choices[0].delta.content == "hi"

    c2 = translate_stream_line(
        {
            "message": {
                "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]
            }
        },
        "m",
        tool_idx,
    )
    tc = c2[0].choices[0].delta.tool_calls[0]
    assert tc.function.name == "f"
    assert json.loads(tc.function.arguments) == {"x": 1}
    assert tc.index == 0
    assert tool_idx["n"] == 1

    # A tool call was streamed, so the terminal chunk's finish_reason must be
    # "tool_calls" even though Ollama reports done_reason="stop".
    c3 = translate_stream_line(
        {"done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2},
        "m",
        tool_idx,
    )
    assert c3[-1].choices[0].finish_reason == "tool_calls"
    assert c3[-1].usage.prompt_tokens == 3


def test_stream_done_without_tool_calls_is_stop():
    # Pure content stream (fresh counter, no tool calls) → finish_reason "stop".
    chunks = translate_stream_line(
        {"done": True, "done_reason": "stop", "prompt_eval_count": 4, "eval_count": 1},
        "m",
        {"n": 0},
    )
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.completion_tokens == 1


def test_embeddings_translation():
    out = _translate_embeddings(
        {
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2]],
            "prompt_eval_count": 4,
        },
        model="nomic-embed-text",
    )
    assert out.object == "list"
    assert out.data[0].embedding == [0.1, 0.2]
    assert out.usage.prompt_tokens == 4


# ── detection ─────────────────────────────────────────────────────────────────


def test_native_root_strips_v1():
    assert native_root("http://h:11434/v1") == "http://h:11434"
    assert native_root("http://h:11434/v1/") == "http://h:11434"
    assert native_root("http://h:11434/") == "http://h:11434"


def test_detection_is_noop_when_flag_unset():
    _OLLAMA_PROBE_CACHE.clear()
    with patch.dict(os.environ, {"HERMES_OLLAMA_NATIVE": ""}, clear=False):
        # No network touched when the flag is off.
        assert is_native_ollama_base_url("http://h:11434/v1") is False


def test_detection_positive_for_ollama():
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(200, json={"version": "0.30.0"})
    with patch.dict(os.environ, {"HERMES_OLLAMA_NATIVE": "1"}, clear=False):
        with patch("agent.ollama_native_adapter.httpx.get", return_value=resp):
            assert is_native_ollama_base_url("http://h:11434/v1") is True


def test_detection_negative_for_non_ollama():
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(404)
    with patch.dict(os.environ, {"HERMES_OLLAMA_NATIVE": "1"}, clear=False):
        with patch("agent.ollama_native_adapter.httpx.get", return_value=resp):
            # A non-Ollama "custom" endpoint (vLLM/llama.cpp/LM Studio) is left on /v1.
            assert is_native_ollama_base_url("http://other:8000/v1") is False


def test_detection_rejects_200_without_version_field():
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(200, json={"status": "ok"})
    with patch.dict(os.environ, {"HERMES_OLLAMA_NATIVE": "1"}, clear=False):
        with patch("agent.ollama_native_adapter.httpx.get", return_value=resp):
            assert is_native_ollama_base_url("http://h:11434") is False


# ── client facade (httpx.MockTransport) ───────────────────────────────────────


def test_chat_completion_posts_num_ctx_to_api_chat():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:e2b",
                "message": {"role": "assistant", "content": "hello"},
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )

    client = OllamaNativeClient(
        base_url="http://h:11434/v1", http_client=_mock_client(handler)
    )
    result = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"options": {"num_ctx": 64000}},
    )
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["options"]["num_ctx"] == 64000
    assert result.choices[0].message.content == "hello"
    assert result.usage.prompt_tokens == 11


def test_streaming_yields_openai_chunks():
    ndjson = (
        '{"message":{"content":"he"}}\n'
        '{"message":{"content":"llo"}}\n'
        '{"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}\n'
    )
    client = OllamaNativeClient(
        base_url="http://h:11434",
        http_client=_mock_client(lambda request: httpx.Response(200, text=ndjson)),
    )
    chunks = list(
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    )
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_headers_drop_non_string_sentinels():
    class _Sentinel:
        pass

    headers = OllamaNativeClient(
        base_url="http://h:11434",
        default_headers={
            "X-Real": "keep",
            "OpenAI-Organization": _Sentinel(),
            "X-None": None,
        },
    )._headers()
    assert headers["X-Real"] == "keep"
    assert "OpenAI-Organization" not in headers
    assert "X-None" not in headers
    assert all(isinstance(v, (str, bytes)) for v in headers.values())


def test_async_client_roundtrip():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "async hi"},
                "done_reason": "stop",
                "prompt_eval_count": 2,
                "eval_count": 1,
            },
        )

    aclient = AsyncOllamaNativeClient(
        OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    )
    result = asyncio.run(
        aclient.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
    )
    assert result.choices[0].message.content == "async hi"
    assert aclient.base_url == "http://h:11434"
