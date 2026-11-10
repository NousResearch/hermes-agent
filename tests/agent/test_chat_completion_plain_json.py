"""Regression: a ``stream=True`` request answered with a plain JSON body.

Field failure (October 2026): an OpenAI-compatible router that ignores
``stream=True`` and returns one plain JSON completion
(``Content-Type: application/json``) made every streaming turn fail with
``EmptyStreamError`` even though the provider had served a complete
response — the SDK returns a stream object for the JSON body, iterating
it yields zero chunks, and the completed-response handling that exists
for adapter shims never engages because the object has no ``choices``.
The only workaround was a per-profile ``model.streaming: false``.

These tests drive the real client paths (main chat wire + auxiliary
stream seams) against an in-process transport that answers streaming
requests with plain JSON, plus an SSE control that must keep working
exactly as before.
"""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from agent.auxiliary_client import _acreate_with_stream, _create_with_progress_once
from agent.chat_completion_helpers import _StreamingCall

MARKER = "PLAIN-JSON-MARKER-42"

COMPLETION = {
    "id": "chatcmpl-plain-json",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "coding",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": MARKER},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

MESSAGES = [{"role": "user", "content": "say the marker"}]


def _json_handler(request: httpx.Request) -> httpx.Response:
    # The endpoint ignores the request's stream flag entirely.
    return httpx.Response(200, json=COMPLETION)


def _sse_handler(request: httpx.Request) -> httpx.Response:
    def chunk(content, finish=None, usage=None):
        payload = {
            "id": "chatcmpl-sse", "object": "chat.completion.chunk",
            "created": 1700000000, "model": "coding",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"

    body = (
        chunk("SSE-")
        + chunk("MARKER", finish="stop", usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        + "data: [DONE]\n\n"
    )
    return httpx.Response(200, content=body.encode(), headers={"Content-Type": "text/event-stream"})


def _client(handler) -> OpenAI:
    return OpenAI(
        base_url="http://router.test/v1", api_key="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )


def _async_client(handler) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="http://router.test/v1", api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False),
    )


class _StubAgent:
    """The slice of AIAgent the streaming call reads, wired to a real SDK client."""

    def __init__(self, client):
        self.model = "coding"
        self.provider = "custom"
        self.base_url = "http://router.test/v1"
        self.api_mode = "chat_completions"
        self.session_id = ""
        self._interrupt_requested = False
        self._disable_streaming = False
        self.log_prefix = ""
        self._client = client
        self.streamed_text: list[str] = []

    def _create_request_openai_client(self, *, reason, api_kwargs):
        return self._client

    def _touch_activity(self, *args):
        pass

    def _capture_rate_limits(self, response):
        pass

    def _capture_credits(self, response):
        pass

    def _capture_nous_model_switch(self, response):
        pass

    def _stream_diag_init(self):
        return {}

    def _stream_diag_capture_response(self, diag, response):
        pass

    def _check_openrouter_cache_status(self, response):
        pass

    def _fire_stream_delta(self, text):
        self.streamed_text.append(text)

    def _abort_request_openai_client(self, client, *, reason):
        pass


def _run_streaming_call(agent) -> object:
    call = _StreamingCall(agent, {"model": "coding", "messages": MESSAGES}, on_first_delta=None)
    return call._call_chat_completions(call._start_stream_attempt())


# ── the incident, replayed on the main chat wire ──────────────────────


def test_main_stream_accepts_plain_json_completion():
    agent = _StubAgent(_client(_json_handler))
    response = _run_streaming_call(agent)
    assert response.choices[0].message.content == MARKER
    # Adopting a completed response switches the session to non-streaming,
    # the same contract as an adapter-returned final response.
    assert agent._disable_streaming is True


def test_main_stream_still_consumes_sse():
    agent = _StubAgent(_client(_sse_handler))
    response = _run_streaming_call(agent)
    assert response.choices[0].message.content == "SSE-MARKER"
    assert agent._disable_streaming is False


# ── the same fallback on the auxiliary stream seams ───────────────────


def test_aux_stream_accepts_plain_json_completion():
    response = _create_with_progress_once(
        _client(_json_handler), {"model": "coding", "messages": MESSAGES}, "test", force_stream=True)
    assert response.choices[0].message.content == MARKER


def test_aux_async_stream_accepts_plain_json_completion():
    async def run():
        return await _acreate_with_stream(
            _async_client(_json_handler), {"model": "coding", "messages": MESSAGES})

    response = asyncio.run(run())
    assert response.choices[0].message.content == MARKER


# ── the converter itself: narrow by construction ──────────────────────


def test_converter_ignores_json_without_choices():
    from agent.chat_completion_plain_json import completed_response_or_stream

    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "nope", "code": "bad_request"}})

    stream = _client(error_handler).chat.completions.create(
        model="coding", messages=MESSAGES, stream=True)
    try:
        # A JSON body that is not a completion stays a stream: the existing
        # empty-stream error path keeps owning that failure.
        assert completed_response_or_stream(stream) is stream
    finally:
        stream.close()


def test_converter_passes_non_sdk_objects_through():
    from agent.chat_completion_plain_json import completed_response_or_stream

    sentinel = object()
    assert completed_response_or_stream(sentinel) is sentinel
