"""Tests for the native Google AI Studio Gemini adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


class DummyResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def test_build_native_request_preserves_thought_signature_on_tool_replay():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[
            {"role": "system", "content": "Be helpful."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                        "extra_content": {
                            "google": {"thought_signature": "sig-123"}
                        },
                    }
                ],
            },
        ],
        tools=[],
        tool_choice=None,
    )

    parts = request["contents"][0]["parts"]
    assert parts[0]["functionCall"]["name"] == "get_weather"
    assert parts[0]["thoughtSignature"] == "sig-123"


def test_build_native_request_uses_original_function_name_for_tool_result():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"forecast": "sunny"}',
            },
        ],
        tools=[],
        tool_choice=None,
    )

    tool_response = request["contents"][1]["parts"][0]["functionResponse"]
    assert tool_response["name"] == "get_weather"


def test_build_native_request_strips_json_schema_only_fields_from_tool_parameters():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Weather lookup",
                    "parameters": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "city": {
                                "type": "string",
                                "$schema": "ignored",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice=None,
    )

    params = request["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "$schema" not in params
    assert "additionalProperties" not in params
    assert params["type"] == "object"
    assert params["properties"]["city"] == {
        "type": "string",
        "description": "City name",
    }


def test_translate_native_response_surfaces_reasoning_and_tool_calls():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "thinking..."},
                        {"functionCall": {"name": "search", "args": {"q": "hermes"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.reasoning == "thinking..."
    assert choice.message.tool_calls[0].function.name == "search"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {"q": "hermes"}


def test_native_client_uses_x_goog_api_key_and_native_models_endpoint(monkeypatch):
    from agent.gemini_native_adapter import GeminiNativeClient

    recorded = {}

    class DummyHTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["json"] = json
            recorded["headers"] = headers
            return DummyResponse(
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "hello"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }
            )

        def close(self):
            return None

    monkeypatch.setattr("agent.gemini_native_adapter.httpx.Client", lambda *a, **k: DummyHTTP())

    client = GeminiNativeClient(api_key="AIza-test", base_url="https://generativelanguage.googleapis.com/v1beta")
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert recorded["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert recorded["headers"]["x-goog-api-key"] == "AIza-test"
    assert "Authorization" not in recorded["headers"]
    assert response.choices[0].message.content == "hello"


def test_native_http_error_keeps_status_and_retry_after():
    from agent.gemini_native_adapter import gemini_http_error

    response = DummyResponse(
        status_code=429,
        headers={"Retry-After": "17"},
        payload={
            "error": {
                "code": 429,
                "message": "quota exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "RESOURCE_EXHAUSTED",
                        "metadata": {"service": "generativelanguage.googleapis.com"},
                    }
                ],
            }
        },
    )

    err = gemini_http_error(response)
    assert getattr(err, "status_code", None) == 429
    assert getattr(err, "retry_after", None) == 17.0
    assert "quota exhausted" in str(err)


def test_native_client_accepts_injected_http_client():
    from agent.gemini_native_adapter import GeminiNativeClient

    injected = SimpleNamespace(close=lambda: None)
    client = GeminiNativeClient(api_key="AIza-test", http_client=injected)
    assert client._http is injected


def test_native_client_rejects_empty_api_key_with_actionable_message():
    """Empty/whitespace api_key must raise at construction, not produce a cryptic
    Google GFE 'Error 400 (Bad Request)!!1' HTML page on the first request."""
    from agent.gemini_native_adapter import GeminiNativeClient

    for bad in ("", "   ", None):
        with pytest.raises(RuntimeError) as excinfo:
            GeminiNativeClient(api_key=bad)  # type: ignore[arg-type]
        msg = str(excinfo.value)
        assert "GOOGLE_API_KEY" in msg and "GEMINI_API_KEY" in msg
        assert "aistudio.google.com" in msg


@pytest.mark.asyncio
async def test_async_native_client_streams_without_requiring_async_iterator_from_sync_client():
    from agent.gemini_native_adapter import AsyncGeminiNativeClient

    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"), finish_reason=None)])
    sync_stream = iter([chunk])

    def _advance(iterator):
        try:
            return False, next(iterator)
        except StopIteration:
            return True, None

    sync_client = SimpleNamespace(
        api_key="AIza-test",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: sync_stream)),
        _advance_stream_iterator=_advance,
        close=lambda: None,
    )

    async_client = AsyncGeminiNativeClient(sync_client)
    stream = await async_client.chat.completions.create(stream=True)
    collected = []
    async for item in stream:
        collected.append(item)
    assert collected == [chunk]


def test_stream_event_translation_emits_tool_call_delta_with_stable_index():
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "abc"}}}
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    first = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    second = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)

    def _tool_call_chunks(chunks):
        return [
            tc
            for chunk in chunks
            for tc in (chunk.choices[0].delta.tool_calls or [])
        ]

    first_tcs = _tool_call_chunks(first)
    second_tcs = _tool_call_chunks(second)

    # Replay of the same event must not re-open or re-emit args for the slot:
    # the consumer already received a full picture from the first translation.
    assert second_tcs == []

    # Indices and ids are stable across emissions of the slot.
    assert {tc.index for tc in first_tcs} == {0}
    ids = {tc.id for tc in first_tcs}
    assert len(ids) == 1

    # Concatenating arguments per OpenAI streaming protocol reconstructs the full args.
    assert _collect_arguments_for_index(first, index=0) == '{"q": "abc"}'

    # Finish chunk is present in both translations of an event carrying finishReason,
    # and signals tool_calls when any slot is open.
    assert first[-1].choices[0].finish_reason == "tool_calls"
    assert second[-1].choices[0].finish_reason == "tool_calls"


def test_stream_event_translation_keeps_identical_calls_in_distinct_parts():
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "abc"}}},
                        {"functionCall": {"name": "search", "args": {"q": "abc"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    tool_chunks = [chunk for chunk in chunks if chunk.choices[0].delta.tool_calls]
    assert tool_chunks[0].choices[0].delta.tool_calls[0].index == 0
    assert tool_chunks[1].choices[0].delta.tool_calls[0].index == 1
    assert tool_chunks[0].choices[0].delta.tool_calls[0].id != tool_chunks[1].choices[0].delta.tool_calls[0].id


def _collect_arguments_for_index(chunks, index):
    """OpenAI streaming protocol: client concatenates delta.tool_calls[].function.arguments
    fragments per index to reconstruct the full JSON args string."""
    fragments = []
    for chunk in chunks:
        deltas = chunk.choices[0].delta.tool_calls or []
        for tc in deltas:
            if getattr(tc, "index", None) == index:
                fragments.append(getattr(tc.function, "arguments", "") or "")
    return "".join(fragments)


def test_stream_event_translation_handles_non_prefix_args_change():
    """Bug repro: when Gemini re-emits a functionCall across SSE events with
    args that are NOT a prefix extension of the previous version, the adapter
    must not produce concatenated fragments that yield invalid JSON.

    OpenAI streaming clients concatenate `delta.tool_calls[].function.arguments`
    by index. Emitting full `{"q":"AB"}` after full `{"q":"A"}` yields
    `{"q":"A"}{"q":"AB"}` — invalid JSON, crashing the consumer.
    """
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event_one = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "A"}}}
                    ]
                },
            }
        ]
    }
    event_two = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "AB"}}}
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    chunks_one = translate_stream_event(
        event_one, model="gemini-2.5-flash", tool_call_indices=tool_call_indices
    )
    chunks_two = translate_stream_event(
        event_two, model="gemini-2.5-flash", tool_call_indices=tool_call_indices
    )

    all_chunks = chunks_one + chunks_two
    args_concat = _collect_arguments_for_index(all_chunks, index=0)

    # Primary protocol guarantee: concat must be valid JSON.
    parsed = json.loads(args_concat)

    # Slot must remain a single tool_call (no spurious split into two index slots).
    indices_seen = {
        tc.index
        for chunk in all_chunks
        for tc in (chunk.choices[0].delta.tool_calls or [])
    }
    assert indices_seen == {0}, f"expected single slot, got {indices_seen}"

    # Final args should be one of the emitted versions (either keep first or last —
    # implementation choice, but must be deterministic and valid).
    assert parsed in ({"q": "A"}, {"q": "AB"})


def test_stream_event_translation_handles_prefix_grow_args():
    """Gemini may stream tool_call args token-by-token (cumulative prefix).
    Adapter should emit only the delta suffix so concat yields the final args.
    """
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event_one = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "abc"}}}
                    ]
                },
            }
        ]
    }
    event_two = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "search",
                                "args": {"q": "abcdef"},
                            }
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    chunks_one = translate_stream_event(
        event_one, model="gemini-2.5-flash", tool_call_indices=tool_call_indices
    )
    chunks_two = translate_stream_event(
        event_two, model="gemini-2.5-flash", tool_call_indices=tool_call_indices
    )

    args_concat = _collect_arguments_for_index(chunks_one + chunks_two, index=0)
    assert json.loads(args_concat) == {"q": "abcdef"}
