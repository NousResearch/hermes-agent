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










# ---------------------------------------------------------------------------
# X-Goog-Api-Client header tests
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Parallel function call slot tests
# ---------------------------------------------------------------------------


def _fc_event(*calls):
    """Build a native Gemini SSE event carrying one functionCall part per call."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": name, "args": args}}
                        for name, args in calls
                    ]
                }
            }
        ]
    }


def _accumulate(events):
    """Replay events through translate_stream_event the way the streaming loop
    in ``_stream_completion`` does, and return {index: concatenated arguments}."""
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices: dict = {}
    acc: dict = {}
    for event in events:
        for chunk in translate_stream_event(
            event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices
        ):
            delta = chunk.choices[0].delta.tool_calls[0]
            acc[delta.index] = acc.get(delta.index, "") + (delta.function.arguments or "")
    return acc


def test_same_tool_called_twice_across_events_gets_distinct_slots():
    """``call_key`` is built from ``part_index``, which restarts at 0 on every
    stream event. Two *different* calls to the same tool arriving in separate
    events therefore hash to the same key and used to share one slot, so their
    arguments were emitted under the same index and concatenated downstream
    into unparseable JSON (`{"query": "A"}{"query": "B"}`)."""
    import json

    acc = _accumulate(
        [
            _fc_event(("web_search", {"query": "A"})),
            _fc_event(("web_search", {"query": "B"})),
        ]
    )

    assert len(acc) == 2, acc
    assert sorted(json.loads(v)["query"] for v in acc.values()) == ["A", "B"]


def test_three_calls_to_same_tool_across_events_each_get_a_slot():
    """The collision compounds: every extra call lands in the same slot."""
    import json

    acc = _accumulate(
        [
            _fc_event(("write_file", {"path": "a"})),
            _fc_event(("write_file", {"path": "b"})),
            _fc_event(("write_file", {"path": "c"})),
        ]
    )

    assert len(acc) == 3, acc
    assert sorted(json.loads(v)["path"] for v in acc.values()) == ["a", "b", "c"]


def test_parallel_calls_in_one_event_keep_working():
    """Regression guard: same-event parallel calls already worked, because each
    part gets its own ``part_index``. This is also the tell that the collision
    is Hermes-side — if the model were concatenating, this would fail too."""
    acc = _accumulate([_fc_event(("web_search", {"query": "A"}), ("web_search", {"query": "B"}))])

    assert len(acc) == 2, acc


def test_different_tools_across_events_keep_working():
    """Regression guard: distinct tool names never collided, since ``name`` is
    part of ``call_key``."""
    acc = _accumulate(
        [
            _fc_event(("read_file", {"path": "a"})),
            _fc_event(("write_file", {"path": "b"})),
        ]
    )

    assert len(acc) == 2, acc


def test_identical_resend_is_still_deduplicated_into_one_slot():
    """Regression guard for the existing dedup path: an identical resend of the
    same part is the same call, not a new one, and must not open a slot."""
    acc = _accumulate(
        [
            _fc_event(("web_search", {"query": "A"})),
            _fc_event(("web_search", {"query": "A"})),
        ]
    )

    assert len(acc) == 1, acc
    assert acc[0] == '{"query": "A"}'


def test_resend_of_the_second_call_reuses_its_collision_created_slot():
    """A slot opened by the collision must stay reachable. Replaying ``[A, B,
    B]``, the resent B starts its lookup from the shared key, whose arguments
    are A's, so it has to be matched against the slot B already opened instead
    of opening a third one and duplicating the call."""
    import json

    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices: dict = {}
    deltas = []
    for event in [
        _fc_event(("web_search", {"query": "A"})),
        _fc_event(("web_search", {"query": "B"})),
        _fc_event(("web_search", {"query": "B"})),
    ]:
        for chunk in translate_stream_event(
            event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices
        ):
            deltas.append(chunk.choices[0].delta.tool_calls[0])

    assert len(tool_call_indices) == 2, tool_call_indices
    assert [d.index for d in deltas] == [0, 1, 1]
    # The resend carries no new arguments and keeps the id of the call it repeats.
    assert deltas[2].function.arguments == ""
    assert deltas[2].id == deltas[1].id
    assert [json.loads(d.function.arguments)["query"] for d in deltas[:2]] == ["A", "B"]


def test_partial_json_arguments_keep_accumulating_in_one_slot():
    """The `json.loads` guard is what keeps a genuinely partial argument string
    in its slot: a half-sent object does not parse, so it is treated as a
    continuation rather than as a new call. The native path always serializes a
    complete dict, so this exercises the guard directly on the accumulator
    state to keep the protective behaviour pinned."""
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices: dict = {}
    translate_stream_event(
        _fc_event(("search", {"q": "A"})),
        model="gemini-2.5-flash",
        tool_call_indices=tool_call_indices,
    )
    # Simulate an incomplete payload already sitting in the slot.
    slot = next(iter(tool_call_indices.values()))
    slot["last_arguments"] = '{"q": "A'

    translate_stream_event(
        _fc_event(("search", {"q": "AB"})),
        model="gemini-2.5-flash",
        tool_call_indices=tool_call_indices,
    )

    assert len(tool_call_indices) == 1, "incomplete JSON must not open a new slot"
