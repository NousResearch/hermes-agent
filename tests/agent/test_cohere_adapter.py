"""Tests for the Cohere adapter helpers (streaming + finish-reason map)."""

from __future__ import annotations

from agent.cohere_adapter import (
    _cohere_finish_reason_to_openai,
    is_cohere_url,
    stream_chat_with_callbacks,
)


# ── is_cohere_url ───────────────────────────────────────────────────────


class TestIsCohereUrl:
    def test_official_host(self):
        assert is_cohere_url("https://api.cohere.com/v2") is True

    def test_legacy_host(self):
        assert is_cohere_url("https://api.cohere.ai/v1") is True

    def test_other_host(self):
        assert is_cohere_url("https://api.openai.com/v1") is False

    def test_empty_and_none(self):
        assert is_cohere_url("") is False
        assert is_cohere_url(None) is False


# ── finish_reason map ──────────────────────────────────────────────────


class TestFinishReasonMap:
    def test_complete(self):
        assert _cohere_finish_reason_to_openai("COMPLETE") == "stop"

    def test_max_tokens(self):
        assert _cohere_finish_reason_to_openai("MAX_TOKENS") == "length"

    def test_tool_call(self):
        assert _cohere_finish_reason_to_openai("TOOL_CALL") == "tool_calls"

    def test_error_toxic(self):
        assert _cohere_finish_reason_to_openai("ERROR_TOXIC") == "content_filter"

    def test_lowercase_input(self):
        assert _cohere_finish_reason_to_openai("complete") == "stop"

    def test_empty(self):
        assert _cohere_finish_reason_to_openai("") == "stop"


# ── stream pump ─────────────────────────────────────────────────────────


def _ev(t: str, **delta_message_fields):
    """Helper: build a dict-shaped stream event for the pump."""
    return {
        "type": t,
        "delta": {"message": delta_message_fields} if delta_message_fields else {},
    }


class TestStreamChatWithCallbacks:
    def test_text_only_stream(self):
        events = [
            {"type": "message-start", "id": "msg_1"},
            _ev("content-delta", content={"text": "Hello"}),
            _ev("content-delta", content={"text": " world"}),
            {
                "type": "message-end",
                "delta": {
                    "finish_reason": "COMPLETE",
                    "usage": {"tokens": {"input_tokens": 5, "output_tokens": 2}},
                },
            },
        ]
        texts = []
        out = stream_chat_with_callbacks(events, on_text_delta=texts.append)
        assert texts == ["Hello", " world"]
        assert out.choices[0].message.content == "Hello world"
        assert out.choices[0].finish_reason == "stop"
        assert out.usage.prompt_tokens == 5
        assert out.usage.completion_tokens == 2

    def test_tool_call_stream_with_tool_plan(self):
        events = [
            _ev("tool-plan-delta", tool_plan="I should "),
            _ev("tool-plan-delta", tool_plan="check the weather."),
            _ev(
                "tool-call-start",
                tool_calls={
                    "id": "tc_1",
                    "function": {"name": "get_weather", "arguments": ""},
                },
            ),
            _ev(
                "tool-call-delta",
                tool_calls={"function": {"arguments": '{"city":'}},
            ),
            _ev(
                "tool-call-delta",
                tool_calls={"function": {"arguments": ' "Tokyo"}'}},
            ),
            _ev("tool-call-end"),
            {
                "type": "message-end",
                "delta": {
                    "finish_reason": "TOOL_CALL",
                    "usage": {"tokens": {"input_tokens": 8, "output_tokens": 3}},
                },
            },
        ]
        tool_starts = []
        reasoning = []
        text_deltas = []
        out = stream_chat_with_callbacks(
            events,
            on_text_delta=text_deltas.append,
            on_tool_start=tool_starts.append,
            on_reasoning_delta=reasoning.append,
        )
        assert tool_starts == ["get_weather"]
        assert "".join(reasoning) == "I should check the weather."
        # No text deltas fire because a tool call is in flight
        assert text_deltas == []
        assert out.choices[0].finish_reason == "tool_calls"
        tc = out.choices[0].message.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"city": "Tokyo"}'

    def test_interrupt_aborts_stream(self):
        # Build an infinite stream-like list that the interrupt callback
        # should stop processing after the first event.
        events = [
            _ev("content-delta", content={"text": "first"}),
            _ev("content-delta", content={"text": "second"}),
        ]
        flag = {"interrupt": False}

        def _interrupted():
            interrupted = flag["interrupt"]
            flag["interrupt"] = True
            return interrupted

        out = stream_chat_with_callbacks(events, on_interrupt_check=_interrupted)
        # First call returns False so the first event processes; the
        # second call returns True and the loop breaks before the second
        # event is consumed.
        assert out.choices[0].message.content == "first"

    def test_partial_tool_call_flushed_at_end(self):
        """If the stream ends mid tool-call (no tool-call-end), the
        adapter should still flush the partial state instead of dropping it."""
        events = [
            _ev(
                "tool-call-start",
                tool_calls={
                    "id": "tc_1",
                    "function": {"name": "do_thing", "arguments": '{"k": "v"}'},
                },
            ),
            # No tool-call-end here.
        ]
        out = stream_chat_with_callbacks(events)
        tcs = out.choices[0].message.tool_calls
        assert tcs is not None and len(tcs) == 1
        assert tcs[0].function.name == "do_thing"
        assert tcs[0].function.arguments == '{"k": "v"}'

    def test_citation_event_fires_callback(self):
        citation = {"start": 0, "end": 5, "text": "hello", "sources": ["doc-1"]}
        events = [
            _ev("content-delta", content={"text": "hello world"}),
            _ev("citation-start", citations=citation),
            {"type": "message-end", "delta": {"finish_reason": "COMPLETE"}},
        ]
        seen = []
        out = stream_chat_with_callbacks(events, on_citation=seen.append)
        assert seen == [citation]
        assert out.choices[0].finish_reason == "stop"
