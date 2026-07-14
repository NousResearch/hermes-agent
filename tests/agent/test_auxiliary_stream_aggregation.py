"""Tests for _aggregate_stream_response and streaming detection in _validate_llm_response.

Covers the fix for custom providers that return SSE streams even when
stream=False is not honored (e.g. title_generation returning raw SSE data).
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_chunk(content="", finish_reason=None, reasoning=""):
    """Build a minimal ChatCompletionChunk-like object."""
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        id="chatcmpl-test123",
        model="test-model",
    )


class TestAggregateStreamResponse:
    """Test _aggregate_stream_response aggregates chunks into completion shape."""

    def test_aggregates_openai_sdk_chunks(self):
        from agent.auxiliary_client import _aggregate_stream_response

        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" "),
            _make_chunk(content="world"),
            _make_chunk(content="", finish_reason="stop"),
        ]
        result = _aggregate_stream_response(iter(chunks), task="title_generation")

        assert hasattr(result, "choices")
        assert len(result.choices) == 1
        msg = result.choices[0].message
        assert msg.content == "Hello world"
        assert msg.role == "assistant"
        assert result.choices[0].finish_reason == "stop"

    def test_aggregates_raw_sse_strings(self):
        from agent.auxiliary_client import _aggregate_stream_response

        lines = [
            f'data: {json.dumps({"id": "cmpl-1", "model": "m", "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]})}',
            f'data: {json.dumps({"id": "cmpl-1", "model": "m", "choices": [{"delta": {"content": " there"}, "finish_reason": None}]})}',
            f'data: {json.dumps({"id": "cmpl-1", "model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}]})}',
            "data: [DONE]",
        ]
        result = _aggregate_stream_response(iter(lines), task="compression")

        assert result.choices[0].message.content == "Hi there"
        assert result.choices[0].finish_reason == "stop"
        assert result.id == "cmpl-1"
        assert result.model == "m"

    def test_preserves_reasoning_content(self):
        from agent.auxiliary_client import _aggregate_stream_response

        chunks = [
            _make_chunk(content="", reasoning="thinking..."),
            _make_chunk(content="answer", reasoning=" more thought"),
            _make_chunk(content="", finish_reason="stop"),
        ]
        result = _aggregate_stream_response(iter(chunks), task="vision")

        assert result.choices[0].message.content == "answer"
        assert result.choices[0].message.reasoning == "thinking... more thought"

    def test_handles_empty_stream(self):
        from agent.auxiliary_client import _aggregate_stream_response

        result = _aggregate_stream_response(iter([]), task="test")

        assert result.choices[0].message.content == ""
        assert result.choices[0].finish_reason == "stop"

    def test_handles_malformed_json_in_sse(self):
        from agent.auxiliary_client import _aggregate_stream_response

        lines = [
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            "data: not-valid-json",
            "data: [DONE]",
        ]
        result = _aggregate_stream_response(iter(lines), task="test")

        # Should accumulate what it can, including the raw fallback text
        content = result.choices[0].message.content
        assert "ok" in content

    def test_handles_iteration_error_gracefully(self):
        from agent.auxiliary_client import _aggregate_stream_response

        def bad_stream():
            yield _make_chunk(content="partial")
            raise ConnectionError("connection dropped")

        result = _aggregate_stream_response(bad_stream(), task="test")

        # Should still return what was collected before the error
        assert result.choices[0].message.content == "partial"


class TestValidateLlmResponseStreamingDetection:
    """Test that _validate_llm_response detects and aggregates streaming responses."""

    def test_passes_normal_completion_through(self):
        from agent.auxiliary_client import _validate_llm_response

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
        )
        result = _validate_llm_response(response, task="test")
        assert result is response

    def test_aggregates_generator_without_choices(self):
        from agent.auxiliary_client import _validate_llm_response

        chunks = [
            _make_chunk(content="aggregated"),
            _make_chunk(content="", finish_reason="stop"),
        ]
        result = _validate_llm_response(iter(chunks), task="title_generation")

        assert hasattr(result, "choices")
        assert result.choices[0].message.content == "aggregated"

    def test_rejects_none(self):
        from agent.auxiliary_client import _validate_llm_response

        with pytest.raises(RuntimeError, match="None response"):
            _validate_llm_response(None, task="test")

    def test_rejects_plain_string_response(self):
        from agent.auxiliary_client import _validate_llm_response

        # Plain strings without "data:" are NOT treated as SSE streams
        with pytest.raises(RuntimeError, match="invalid response"):
            _validate_llm_response("raw string response", task="test")

    def test_aggregates_sse_string_response(self):
        from agent.auxiliary_client import _validate_llm_response

        # Strings containing SSE "data:" lines ARE aggregated as streams.
        # This is the exact case from the bug report: custom provider returns
        # type=str with raw SSE content instead of a parsed completion object.
        sse_blob = (
            'data: {"id":"cmpl-1","model":"m","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n'
            'data: {"id":"cmpl-1","model":"m","choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n'
            "data: [DONE]\n"
        )
        result = _validate_llm_response(sse_blob, task="title_generation")
        assert hasattr(result, "choices")
        assert result.choices[0].message.content == "Hello world"

    def test_rejects_dict_without_choices_attr(self):
        from agent.auxiliary_client import _validate_llm_response

        # Dicts are excluded from stream detection
        with pytest.raises(RuntimeError, match="invalid response"):
            _validate_llm_response({"error": "something"}, task="test")

    def test_rejects_empty_choices(self):
        from agent.auxiliary_client import _validate_llm_response

        response = SimpleNamespace(choices=[])
        with pytest.raises(RuntimeError, match="invalid response"):
            _validate_llm_response(response, task="test")

    def test_rejects_choices_without_message(self):
        from agent.auxiliary_client import _validate_llm_response

        response = SimpleNamespace(choices=[SimpleNamespace(index=0)])
        with pytest.raises(RuntimeError, match="invalid response"):
            _validate_llm_response(response, task="test")
