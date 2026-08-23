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











def test_followup_user_turn_is_not_merged_into_function_response_turn():
    """Human follow-up after tool results must stay its own user content.

    The split pair is kept alternation-valid by interposing a placeholder
    model turn between the functionResponse content and the human text
    content (mirrors gemini-cli#28700's INTERRUPTED_RESPONSE_PLACEHOLDER).

    Scope: only the functionResponse↔human-text boundary. Ordinary same-role
    merges (parallel tool results, back-to-back plain user texts) remain
    required for Gemini alternation and are covered by sibling tests.
    """
    from agent.gemini_native_adapter import (
        _INTERRUPTED_RESPONSE_PLACEHOLDER,
        _build_gemini_contents,
    )

    messages = [
        {"role": "user", "content": "Load the skill"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "skill_view",
                        "arguments": '{"name":"hermes-agent"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "loaded"},
        {"role": "user", "content": "Continue"},
    ]

    contents, _ = _build_gemini_contents(messages)

    assert [content["role"] for content in contents] == [
        "user",
        "model",
        "user",
        "model",
        "user",
    ]
    assert "functionResponse" in contents[2]["parts"][0]
    assert contents[3]["parts"] == [{"text": _INTERRUPTED_RESPONSE_PLACEHOLDER}]
    assert contents[-1]["parts"] == [{"text": "Continue"}]


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


def test_schema_bearing_tool_result_is_wrapped_as_opaque_text():
    """A tool result whose content is itself a JSON Schema must not be
    forwarded as a structured functionResponse.response.

    Gemini 3 resolves ``$ref``/``$defs`` pointers inside a function response
    payload and rejects unknown references with HTTP 400 INVALID_ARGUMENT
    ("referenced name '#/$defs/...' does not match a display_name"; see
    vercel/ai#14369). ``tool_describe`` output for an MCP tool is exactly such
    a schema, so it must be wrapped as opaque text instead.
    """
    from agent.gemini_native_adapter import _translate_tool_result_to_gemini

    schema = {
        "$defs": {"SetCookieParam": {"type": "object"}},
        "properties": {"cookies": {"$ref": "#/$defs/SetCookieParam"}},
    }
    msg = {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "tool_describe",
        "content": json.dumps(schema),
    }

    out = _translate_tool_result_to_gemini(msg, include_ids=True)

    response = out["functionResponse"]["response"]
    assert "$defs" not in response
    assert "output" in response
    # The raw schema text is preserved verbatim in the wrapped output.
    assert "#/$defs/SetCookieParam" in response["output"]


def test_plain_json_tool_result_remains_structured():
    """Ordinary JSON tool results without a ``$ref`` pointer keep the
    structured form (no regression to the existing structured-response path)."""
    from agent.gemini_native_adapter import _translate_tool_result_to_gemini

    msg = {
        "role": "tool",
        "tool_call_id": "call_2",
        "name": "some_tool",
        "content": json.dumps({"status": "ok", "count": 3}),
    }

    out = _translate_tool_result_to_gemini(msg)

    assert out["functionResponse"]["response"] == {"status": "ok", "count": 3}




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


def test_build_gemini_request_preserves_explicit_max_tokens_without_thinking():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4096,
    )

    assert request["generationConfig"]["maxOutputTokens"] == 4096
    assert "thinkingConfig" not in request["generationConfig"]


def test_build_gemini_request_raises_max_output_when_thinking_is_enabled():
    from agent.gemini_native_adapter import (
        GEMINI_DEFAULT_MAX_OUTPUT_TOKENS,
        build_gemini_request,
    )

    request = build_gemini_request(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4096,
        thinking_config={"includeThoughts": True, "thinkingLevel": "high"},
    )

    assert request["generationConfig"]["maxOutputTokens"] == GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    assert request["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


def test_build_gemini_request_does_not_raise_when_thinking_is_disabled():
    from agent.gemini_native_adapter import build_gemini_request

    request = build_gemini_request(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4096,
        thinking_config={"includeThoughts": False},
    )

    assert request["generationConfig"]["maxOutputTokens"] == 4096
    assert request["generationConfig"]["thinkingConfig"]["includeThoughts"] is False










# ---------------------------------------------------------------------------
# X-Goog-Api-Client header tests
# ---------------------------------------------------------------------------










class TestGemini3ToolCallIds:
    """Gemini 3+ requires explicit tool call IDs in replayed history
    (port of earendil-works/pi#7494)."""

    def _history(self):
        return [
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

    def test_requires_ids_gate(self):
        from agent.gemini_native_adapter import gemini_requires_tool_call_ids

        assert gemini_requires_tool_call_ids("gemini-3.6-flash")
        assert gemini_requires_tool_call_ids("google/gemini-3.6-pro")
        assert gemini_requires_tool_call_ids("gemini-3-flash-preview")
        assert not gemini_requires_tool_call_ids("gemini-2.5-flash")
        assert not gemini_requires_tool_call_ids("gemini-1.5-pro")
        assert not gemini_requires_tool_call_ids("claude-opus-4.6")
        assert not gemini_requires_tool_call_ids("")

    def test_ids_preserved_for_gemini3(self):
        from agent.gemini_native_adapter import _build_gemini_contents

        contents, _ = _build_gemini_contents(
            self._history(), include_tool_call_ids=True
        )
        call_ids = [
            p["functionCall"]["id"]
            for c in contents for p in c["parts"] if "functionCall" in p
        ]
        response_ids = [
            p["functionResponse"]["id"]
            for c in contents for p in c["parts"] if "functionResponse" in p
        ]
        assert call_ids == ["call_1", "call_2"]
        assert response_ids == ["call_1", "call_2"]

    def test_ids_omitted_for_older_gemini(self):
        from agent.gemini_native_adapter import _build_gemini_contents

        contents, _ = _build_gemini_contents(self._history())
        for c in contents:
            for p in c["parts"]:
                if "functionCall" in p:
                    assert "id" not in p["functionCall"]
                if "functionResponse" in p:
                    assert "id" not in p["functionResponse"]

    def test_build_request_threads_model_gate(self):
        from agent.gemini_native_adapter import build_gemini_request

        request = build_gemini_request(
            messages=self._history(), model="gemini-3.6-flash"
        )
        parts = [p for c in request["contents"] for p in c["parts"]]
        assert any(p.get("functionCall", {}).get("id") == "call_1" for p in parts)

        request_old = build_gemini_request(
            messages=self._history(), model="gemini-2.5-flash"
        )
        parts_old = [p for c in request_old["contents"] for p in c["parts"]]
        assert all("id" not in p.get("functionCall", {}) for p in parts_old)

    def test_response_preserves_provider_tool_call_id(self):
        from agent.gemini_native_adapter import translate_gemini_response

        resp = {
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"id": "call_native_7", "name": "read_file",
                                     "args": {"path": "a.txt"}},
                }]},
                "finishReason": "STOP",
            }],
        }
        result = translate_gemini_response(resp, model="gemini-3.6-flash")
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0].id == "call_native_7"

    def test_response_generates_id_when_absent(self):
        from agent.gemini_native_adapter import translate_gemini_response

        resp = {
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "read_file", "args": {}},
                }]},
                "finishReason": "STOP",
            }],
        }
        result = translate_gemini_response(resp, model="gemini-2.5-flash")
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0].id.startswith("call_")
