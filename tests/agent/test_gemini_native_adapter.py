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


def test_parallel_tool_results_merge_into_one_user_content():
    """Gemini requires strict user/model alternation; two consecutive `user`
    contents are rejected with HTTP 400. Parallel tool calls produce two tool
    results in a row, so their functionResponses must be grouped into a single
    user content instead of two consecutive ones."""
    from agent.gemini_native_adapter import _build_gemini_contents

    messages = [
        {"role": "user", "content": "Read a.txt and b.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "AAA"},
        {"role": "tool", "tool_call_id": "call_2", "content": "BBB"},
    ]

    contents, _ = _build_gemini_contents(messages)
    roles = [c["role"] for c in contents]

    # No two adjacent contents may share a role.
    assert all(roles[i] != roles[i - 1] for i in range(1, len(roles))), roles
    assert roles == ["user", "model", "user"]

    # Both parallel functionResponses land in the single trailing user content.
    response_parts = [
        p for p in contents[2]["parts"] if "functionResponse" in p
    ]
    outputs = [p["functionResponse"]["response"]["output"] for p in response_parts]
    assert outputs == ["AAA", "BBB"]


def test_consecutive_user_messages_merge_for_gemini_alternation():
    """Back-to-back user messages must also be merged, not sent as two
    consecutive user contents."""
    from agent.gemini_native_adapter import _build_gemini_contents

    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]
    contents, _ = _build_gemini_contents(messages)
    roles = [c["role"] for c in contents]
    assert roles == ["user", "model"], roles


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


@pytest.mark.parametrize("model, expected", [
    ("google/gemini-2.0-flash", "gemini-2.0-flash"),
    ("gemini/gemini-3-pro-preview", "gemini-3-pro-preview"),
    ("Google/Gemini-2.5-Pro", "Gemini-2.5-Pro"),
    ("models/gemini-x", "models/gemini-x"),
    ("tunedModels/my-tune", "tunedModels/my-tune"),
])
def test_bare_gemini_model_id_strips_only_self_prefix(model, expected):
    from agent.gemini_native_adapter import bare_gemini_model_id

    assert bare_gemini_model_id(model) == expected


def test_native_client_strips_self_prefix_from_model_url(monkeypatch):
    from agent.gemini_native_adapter import GeminiNativeClient

    recorded = {}

    class DummyHTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            return DummyResponse(payload={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            })

        def close(self):
            return None

    monkeypatch.setattr("agent.gemini_native_adapter.httpx.Client", lambda *a, **k: DummyHTTP())
    client = GeminiNativeClient(api_key="AIza-test", base_url="https://generativelanguage.googleapis.com/v1beta")
    client.chat.completions.create(
        model="google/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert recorded["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


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

    assert first[0].choices[0].delta.tool_calls[0].index == 0
    assert second[0].choices[0].delta.tool_calls[0].index == 0
    assert first[0].choices[0].delta.tool_calls[0].id == second[0].choices[0].delta.tool_calls[0].id
    assert first[0].choices[0].delta.tool_calls[0].function.arguments == '{"q": "abc"}'
    assert second[0].choices[0].delta.tool_calls[0].function.arguments == ""
    assert first[-1].choices[0].finish_reason == "tool_calls"


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


def test_system_instruction_includes_role_field_and_stays_out_of_contents():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        tools=[],
        tool_choice=None,
    )

    assert request["systemInstruction"] == {
        "role": "system",
        "parts": [{"text": "You are a helpful assistant."}],
    }
    assert all(content.get("role") != "system" for content in request["contents"])


def test_max_tokens_none_defaults_to_gemini_output_ceiling():
    """max_tokens=None must send the model's full output ceiling, not omit it.

    Gemini's native generateContent applies a low internal default when
    maxOutputTokens is absent, truncating tool calls mid-stream. Hermes passes
    None to mean "unlimited", so the adapter must translate that to the
    published 65,535 ceiling rather than leaving the field unset.
    """
    from agent.gemini_native_adapter import (
        build_gemini_request,
        GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
    )

    req = build_gemini_request(messages=[{"role": "user", "content": "hi"}], max_tokens=None)
    assert req["generationConfig"]["maxOutputTokens"] == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS == 65535


def test_explicit_max_tokens_is_respected():
    from agent.gemini_native_adapter import build_gemini_request

    req = build_gemini_request(messages=[{"role": "user", "content": "hi"}], max_tokens=4096)
    assert req["generationConfig"]["maxOutputTokens"] == 4096


def test_stream_event_translation_separates_parallel_calls_across_events():
    """Two distinct parallel calls to the same tool arrive in separate SSE
    events, each with part_index=0 and the same name — they must become two
    tool calls, not one slot with concatenated JSON arguments."""
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event_a = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "read_file", "args": {"path": "A"}}}]}}
        ]
    }
    event_b = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "read_file", "args": {"path": "B"}}}]}},
        ]
    }

    first = translate_stream_event(event_a, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    second = translate_stream_event(event_b, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)

    call_a = first[0].choices[0].delta.tool_calls[0]
    call_b = second[0].choices[0].delta.tool_calls[0]
    assert call_a.index == 0
    assert call_b.index == 1
    assert call_a.id != call_b.id
    assert call_a.function.arguments == '{"path": "A"}'
    assert call_b.function.arguments == '{"path": "B"}'


def test_stream_event_translation_chains_three_parallel_calls_across_events():
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    calls = []
    for path in ("A", "B", "C"):
        event = {
            "candidates": [
                {"content": {"parts": [{"functionCall": {"name": "read_file", "args": {"path": path}}}]}}
            ]
        }
        chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
        calls.append(chunks[0].choices[0].delta.tool_calls[0])

    assert [c.index for c in calls] == [0, 1, 2]
    assert len({c.id for c in calls}) == 3
    assert [c.function.arguments for c in calls] == ['{"path": "A"}', '{"path": "B"}', '{"path": "C"}']


def test_stream_event_translation_still_dedups_resent_identical_call_across_events():
    """Identical re-sends of the same call across events must still dedup
    (emit empty arguments the second time) instead of opening a new slot."""
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "search", "args": {"q": "abc"}}}]}}
        ]
    }

    first = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    second = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)

    assert first[0].choices[0].delta.tool_calls[0].index == 0
    assert second[0].choices[0].delta.tool_calls[0].index == 0
    assert second[0].choices[0].delta.tool_calls[0].function.arguments == ""
