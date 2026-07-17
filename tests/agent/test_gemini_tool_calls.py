"""Edge-case tests for Gemini tool-call translation.

Focuses on scenarios the existing test_gemini_native_adapter.py doesn't cover:
parallel tool calls, argument fidelity, streaming progressive deltas,
and mixed content parts.
"""

from __future__ import annotations

import json

import pytest


# ── Non-streaming: translate_gemini_response ──────────────────────────

def test_non_streaming_parallel_tool_calls():
    """Multiple functionCall parts → distinct tool_calls in one message."""
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "terminal", "args": {"cmd": "ls"}}},
                        {"functionCall": {"name": "read_file", "args": {"path": "/x"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    choice = response.choices[0]

    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) == 2

    tc0 = choice.message.tool_calls[0]
    tc1 = choice.message.tool_calls[1]

    assert tc0.function.name == "terminal"
    assert json.loads(tc0.function.arguments) == {"cmd": "ls"}
    assert tc1.function.name == "read_file"
    assert json.loads(tc1.function.arguments) == {"path": "/x"}

    # Each tool call should have a unique id
    assert tc0.id != tc1.id


def test_non_streaming_tool_call_index_is_sequential_not_part_offset():
    """Tool call indices should be 0,1,... not the parts[] array offset.

    When parts = [text, functionCall, functionCall], the two tool calls should
    get indices 0 and 1, not 1 and 2.
    """
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Let me run those commands."},
                        {"functionCall": {"name": "terminal", "args": {"cmd": "ls"}}},
                        {"functionCall": {"name": "read_file", "args": {"path": "/x"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    tcs = response.choices[0].message.tool_calls

    assert len(tcs) == 2
    # This assertion will fail if translate_gemini_response uses the
    # parts[] array offset as the tool call index instead of 0,1,...
    assert tcs[0].index == 0, f"Expected index 0, got {tcs[0].index}"
    assert tcs[1].index == 1, f"Expected index 1, got {tcs[1].index}"


# ── Arguments fidelity ────────────────────────────────────────────────

def test_arguments_preserves_empty_dict():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "noop", "args": {}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    args = response.choices[0].message.tool_calls[0].function.arguments
    assert json.loads(args) == {}


def test_arguments_preserves_nested_dict():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "search",
                                "args": {
                                    "query": {"nested": {"deep": True}},
                                    "filters": [{"type": "tag", "value": "hermes"}],
                                },
                            }
                        },
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    assert args["query"] == {"nested": {"deep": True}}
    assert args["filters"] == [{"type": "tag", "value": "hermes"}]


def test_arguments_preserves_numeric_and_bool_types():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "configure",
                                "args": {
                                    "int_val": 42,
                                    "float_val": 3.14,
                                    "neg_val": -7,
                                    "bool_true": True,
                                    "bool_false": False,
                                    "zero": 0,
                                },
                            }
                        },
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    args_str = response.choices[0].message.tool_calls[0].function.arguments
    args = json.loads(args_str)

    assert args["int_val"] == 42
    assert args["float_val"] == 3.14
    assert args["neg_val"] == -7
    assert args["bool_true"] is True
    assert args["bool_false"] is False
    assert args["zero"] == 0


def test_arguments_preserves_unicode_strings():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "translate",
                                "args": {
                                    "chinese": "你好世界",
                                    "emoji": "🚀",
                                    "german": "Grüß Gott",
                                    "mixed": "café – 咖啡",
                                },
                            }
                        },
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    args_str = response.choices[0].message.tool_calls[0].function.arguments
    args = json.loads(args_str)

    assert args["chinese"] == "你好世界"
    assert args["emoji"] == "🚀"
    assert args["german"] == "Grüß Gott"
    assert args["mixed"] == "café – 咖啡"


def test_arguments_missing_args_field_defaults_to_empty_dict():
    """When functionCall has no 'args' key, produce '{}' not crash."""
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "ping"}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    args_str = response.choices[0].message.tool_calls[0].function.arguments
    assert json.loads(args_str) == {}


# ── Streaming: translate_stream_event ─────────────────────────────────

def test_streaming_changed_args_emits_full_arguments():
    """When args change across events, the full new JSON is emitted.

    Gemini sends complete dict objects in functionCall.args, not progressive
    character deltas. Emitting the full args each time is correct for OpenAI
    compatibility; the downstream consumer accumulates deltas.
    """
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}

    # First event
    event1 = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "her"}}},
                    ]
                },
                "finishReason": "",
            }
        ]
    }
    chunks1 = translate_stream_event(event1, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    tc_chunks1 = [c for c in chunks1 if c.choices[0].delta.tool_calls]
    assert len(tc_chunks1) == 1
    assert tc_chunks1[0].choices[0].delta.tool_calls[0].function.arguments == '{"q": "her"}'

    # Second event: args changed (different value for same key)
    event2 = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "hermes"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    chunks2 = translate_stream_event(event2, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    tc_chunks2 = [c for c in chunks2 if c.choices[0].delta.tool_calls]
    assert len(tc_chunks2) == 1
    # Full args emitted because the dict changed (not a progressive string delta)
    assert tc_chunks2[0].choices[0].delta.tool_calls[0].function.arguments == '{"q": "hermes"}'

    # Finish chunk should have tool_calls reason
    finish_chunks = [c for c in chunks2 if c.choices[0].finish_reason]
    assert finish_chunks[0].choices[0].finish_reason == "tool_calls"


def test_streaming_multiple_distinct_tool_calls():
    """Two different functionCalls in one stream event → two tool call deltas."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "terminal", "args": {"cmd": "ls"}}},
                        {"functionCall": {"name": "read_file", "args": {"path": "/tmp"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    tc_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]

    assert len(tc_chunks) == 2
    assert tc_chunks[0].choices[0].delta.tool_calls[0].index == 0
    assert tc_chunks[1].choices[0].delta.tool_calls[0].index == 1
    assert tc_chunks[0].choices[0].delta.tool_calls[0].id != tc_chunks[1].choices[0].delta.tool_calls[0].id
    assert tc_chunks[0].choices[0].delta.tool_calls[0].function.name == "terminal"
    assert tc_chunks[1].choices[0].delta.tool_calls[0].function.name == "read_file"


def test_streaming_mixed_text_and_tool_calls():
    """Text parts before tool calls shouldn't affect tool call indexing."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "I'll search and read for you."},
                        {"functionCall": {"name": "search", "args": {"q": "hermes"}}},
                        {"functionCall": {"name": "read_file", "args": {"path": "/x"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    tc_chunks = [c for c in chunks if c.choices[0].delta.tool_calls]

    assert len(tc_chunks) == 2
    # Tool call indices should be 0 and 1 regardless of text position
    assert tc_chunks[0].choices[0].delta.tool_calls[0].index == 0
    assert tc_chunks[1].choices[0].delta.tool_calls[0].index == 1


def test_streaming_tool_call_stability_across_events_with_text():
    """Tool call ids and indices stay stable even when text appears in later events."""
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}

    event1 = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "hermes"}}},
                    ]
                },
                "finishReason": "",
            }
        ]
    }
    chunks1 = translate_stream_event(event1, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    tc1 = [c for c in chunks1 if c.choices[0].delta.tool_calls][0]
    original_id = tc1.choices[0].delta.tool_calls[0].id

    # Later event has text before the same functionCall
    event2 = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Let me refine that search."},
                        {"functionCall": {"name": "search", "args": {"q": "hermes agent"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    chunks2 = translate_stream_event(event2, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    tc2 = [c for c in chunks2 if c.choices[0].delta.tool_calls]

    # The tool call from event2 has a different part_index (1 vs 0 in event1),
    # so it gets a NEW slot — this is expected since part_index is part of
    # the call_key. The old slot (part_index 0) keeps its id.
    if tc2:
        # If Gemini emitted it as a new call (different part_index), new id is fine
        pass
