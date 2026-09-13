"""Chat-completions request-transform bypass (#93650 extended to chat.completions).

``chat.completions.create`` re-walks the whole request body against the
``CompletionCreateParams`` union client-side, GIL held, before any byte leaves
the process. The bypass moves the already-wire-format bulk fields into
``extra_body``; its whole safety argument is that the server receives the same
bytes, so that is what these tests pin.
"""

import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import httpx
import openai

from agent.sdk_transform_bypass import ESCAPE_HATCH_ENV, bypass_chat_sdk_request_transform

_SSE = (
    b'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n"
)


def _wire_body() -> dict:
    """Production-shaped chat body: content parts incl. an image, a tool_calls turn, a tool
    result, function tool schemas, and a caller-populated extra_body (reasoning/provider)."""
    return {
        "model": "hermes-4-70b",
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "https://e.example/i.png", "detail": "low"}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "total 0"},
        ],
        "tools": [
            {"type": "function", "function": {"name": f"tool_{i}", "description": "d",
             "parameters": {"type": "object", "properties": {"p": {"type": "string"}}, "required": ["p"]}}}
            for i in range(3)
        ],
        "tool_choice": "auto",
        "stream": True,
        "temperature": 0.7,
        "stream_options": {"include_usage": True},
        "extra_body": {"reasoning": {"effort": "high"}, "provider": {"order": ["nous"]}},
    }


class _Recorder:
    """A real openai.OpenAI client whose transport records the request bytes."""

    def __init__(self):
        self.content: bytes | None = None
        self.client = openai.OpenAI(
            api_key="k", base_url="https://chat.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.content = request.content
        return httpx.Response(200, content=_SSE, headers={"content-type": "text/event-stream"})

    def send(self, kwargs: dict) -> bytes:
        for _ in self.client.chat.completions.create(**kwargs):
            pass
        assert self.content is not None
        return self.content


def test_bulk_fields_ride_in_extra_body_and_the_wire_bytes_are_identical():
    """Same bytes → same server behaviour and the same byte-keyed prompt-cache prefix."""
    recorder = _Recorder()
    body = _wire_body()

    moved = bypass_chat_sdk_request_transform(dict(body), recorder.client)

    assert moved is not body  # the bypass engaged; the SDK's transform never saw the bulk fields
    assert "messages" in moved["extra_body"]
    assert recorder.send(moved) == recorder.send(dict(body))


def test_escape_hatch_restores_the_typed_sdk_path(monkeypatch):
    monkeypatch.setenv(ESCAPE_HATCH_ENV, "1")
    recorder = _Recorder()
    kwargs = _wire_body()

    assert bypass_chat_sdk_request_transform(kwargs, recorder.client) is kwargs
