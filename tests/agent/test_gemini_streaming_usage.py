"""Tests for Gemini native streaming usageMetadata extraction (issue #15253).

The bug: translate_stream_event() never extracts usageMetadata from SSE events,
so all Gemini streaming sessions show input_tokens=0, output_tokens=0.

The fix: attach usage from usageMetadata to the finish-reason chunk, mirroring
the non-streaming translate_gemini_response().
"""


def test_stream_event_with_usage_metadata_attaches_to_finish_chunk():
    """When a streaming event includes usageMetadata, it should be attached
    to the finish_reason chunk (the last chunk emitted)."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 7,
            "totalTokenCount": 49,
            "cachedContentTokenCount": 10,
        },
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    assert len(chunks) == 2  # content chunk + finish chunk

    # First chunk (content) should have no usage
    assert chunks[0].usage is None

    # Second chunk (finish) should carry usage from the event
    finish_chunk = chunks[-1]
    assert finish_chunk.choices[0].finish_reason == "stop"
    assert finish_chunk.usage is not None
    assert finish_chunk.usage.prompt_tokens == 42
    assert finish_chunk.usage.completion_tokens == 7
    assert finish_chunk.usage.total_tokens == 49
    assert finish_chunk.usage.prompt_tokens_details.cached_tokens == 10


def test_stream_event_with_usage_metadata_and_tool_calls():
    """Usage should be attached to the finish chunk even when tool calls are present."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "totalTokenCount": 120,
        },
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    # Should have tool call chunk + finish chunk
    finish_chunk = chunks[-1]
    assert finish_chunk.choices[0].finish_reason == "tool_calls"
    assert finish_chunk.usage is not None
    assert finish_chunk.usage.prompt_tokens == 100
    assert finish_chunk.usage.completion_tokens == 20
    assert finish_chunk.usage.total_tokens == 120


def test_stream_event_without_usage_metadata_has_no_usage():
    """When usageMetadata is absent (intermediate chunks), usage stays None."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {"parts": [{"text": "thinking"}]},
            }
        ],
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    assert len(chunks) == 1
    assert chunks[0].usage is None


def test_stream_event_with_empty_usage_metadata_has_no_usage():
    """When usageMetadata is present but empty, usage should be None
    (don't attach a zero-usage namespace when Gemini sends an empty dict)."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {},
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    finish_chunk = chunks[-1]
    # Empty usageMetadata dict -> no usage attached (no data to report)
    assert finish_chunk.usage is None


def test_stream_event_usage_with_missing_fields_defaults_to_zero():
    """When usageMetadata is partially present, missing fields default to 0."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "totalTokenCount": 60,
            # candidatesTokenCount and cachedContentTokenCount missing
        },
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    finish_chunk = chunks[-1]
    assert finish_chunk.usage is not None
    assert finish_chunk.usage.prompt_tokens == 50
    assert finish_chunk.usage.completion_tokens == 0  # missing -> 0
    assert finish_chunk.usage.total_tokens == 60
    assert finish_chunk.usage.prompt_tokens_details.cached_tokens == 0  # missing -> 0


def test_stream_event_with_reasoning_and_usage():
    """Usage should appear on finish chunk even when reasoning (thought) parts are present."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "Let me think..."},
                        {"text": "The answer is 42."},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 30,
            "candidatesTokenCount": 15,
            "totalTokenCount": 45,
            "cachedContentTokenCount": 5,
        },
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    # reasoning + content + finish
    assert len(chunks) == 3
    finish_chunk = chunks[-1]
    assert finish_chunk.choices[0].finish_reason == "stop"
    assert finish_chunk.usage is not None
    assert finish_chunk.usage.prompt_tokens == 30
    assert finish_chunk.usage.completion_tokens == 15
    assert finish_chunk.usage.prompt_tokens_details.cached_tokens == 5

def test_stream_event_with_explicit_zero_usage_metadata():
    """When usageMetadata has all zero values, we should still attach it
    (distinct from None -- tells the caller we got usage data)."""
    from agent.gemini_native_adapter import translate_stream_event

    event = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hi"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        },
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    finish_chunk = chunks[-1]
    assert finish_chunk.usage is not None
    assert finish_chunk.usage.prompt_tokens == 0
    assert finish_chunk.usage.completion_tokens == 0
    assert finish_chunk.usage.total_tokens == 0


def test_stream_event_usage_only_on_finish_not_content():
    """usageMetadata is ignored on intermediate chunks (no finishReason).
    Only the finish chunk should carry usage."""
    from agent.gemini_native_adapter import translate_stream_event

    # Intermediate chunk: has usageMetadata but no finishReason
    event = {
        "candidates": [
            {
                "content": {"parts": [{"text": "thinking"}]},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6,
        },
    }

    chunks = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices={})
    # No finish chunk, so no usage attached at all
    for chunk in chunks:
        assert chunk.usage is None
