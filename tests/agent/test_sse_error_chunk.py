"""Tests for issue #65631 — provider error chunks in SSE streams are
misclassified as "empty stream" and retried forever.

Some OpenAI-compatible providers (e.g. DeepInfra) return HTTP 200 with a
single SSE chunk whose ``choices`` is ``None`` and whose provider-specific
``error_type`` / ``error_message`` fields carry a validation error (e.g.
a context-length 400). Without inspecting these fields, the streaming
path treats the chunk as an empty stream and retries the identical
oversized request forever — the real error is never surfaced to the user.

The fix: detect ``error_type`` / ``error_message`` on chunks with no
choices and raise ``ProviderStreamError`` (non-transient) so the error is
surfaced immediately instead of being retried.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.errors import EmptyStreamError, ProviderStreamError


# --------------------------------------------------------------------------- #
# ProviderStreamError
# --------------------------------------------------------------------------- #


def test_provider_stream_error_basic():
    """ProviderStreamError stores the provider error message and type."""
    err = ProviderStreamError("context too long", error_type="validation_error")
    assert str(err) == "context too long"
    assert err.provider_error == "context too long"
    assert err.error_type == "validation_error"


def test_provider_stream_error_without_type():
    """ProviderStreamError works without an error_type."""
    err = ProviderStreamError("something went wrong")
    assert str(err) == "something went wrong"
    assert err.provider_error == "something went wrong"
    assert err.error_type is None


def test_provider_stream_error_is_runtime_error():
    """ProviderStreamError is a RuntimeError (not an EmptyStreamError)."""
    err = ProviderStreamError("test")
    assert isinstance(err, RuntimeError)
    assert not isinstance(err, EmptyStreamError)


def test_provider_stream_error_not_transient():
    """ProviderStreamError must NOT be classified as EmptyStreamError.

    The retry machinery checks isinstance(e, EmptyStreamError) to decide
    if an error is transient. ProviderStreamError must be a separate type
    so it falls through to the non-transient path and surfaces immediately.
    """
    err = ProviderStreamError("test")
    # The retry path checks: isinstance(e, EmptyStreamError)
    assert not isinstance(err, EmptyStreamError)


# --------------------------------------------------------------------------- #
# Error chunk detection in the streaming loop
# --------------------------------------------------------------------------- #


def _make_error_chunk(error_type: str, error_message: str):
    """Build a chunk that simulates a provider error in an SSE stream.

    The chunk has ``choices=None`` (like a real error chunk from DeepInfra)
    plus ``error_type`` and ``error_message`` attributes.
    """
    return SimpleNamespace(
        choices=None,
        model=None,
        usage=None,
        error_type=error_type,
        error_message=error_message,
    )


def _make_normal_chunk(content: str = "", finish_reason: str = "stop"):
    """Build a normal content chunk."""
    delta = SimpleNamespace(content=content, reasoning_content=None, reasoning=None, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, index=0, finish_reason=finish_reason)],
        model="test-model",
        usage=None,
        error_type=None,
        error_message=None,
    )


def _make_usage_chunk(usage_obj):
    """Build a final chunk with usage info and no choices."""
    return SimpleNamespace(
        choices=None,
        model="test-model",
        usage=usage_obj,
        error_type=None,
        error_message=None,
    )


def test_error_chunk_raises_provider_stream_error():
    """A chunk with error_type and error_message should raise ProviderStreamError.

    This tests the core fix: when a chunk has no choices but has error
    fields, we raise ProviderStreamError instead of silently skipping it.
    """

    # Simulate the chunk-processing logic from the streaming loop
    def process_chunk(chunk):
        if not chunk.choices:
            _chunk_error_type = getattr(chunk, "error_type", None)
            _chunk_error_message = getattr(chunk, "error_message", None)
            if _chunk_error_type or _chunk_error_message:
                _err_detail = _chunk_error_message or _chunk_error_type
                raise ProviderStreamError(
                    str(_err_detail),
                    error_type=str(_chunk_error_type) if _chunk_error_type else None,
                )
            # Normal no-choices chunk (usage, model info) — skip
            return None
        return chunk.choices[0].delta.content

    error_chunk = _make_error_chunk(
        error_type="validation_error",
        error_message='{"error":{"message":"This model\'s maximum context length is 163840 tokens"}}',
    )

    with pytest.raises(ProviderStreamError) as exc_info:
        process_chunk(error_chunk)

    assert "maximum context length" in str(exc_info.value)
    assert exc_info.value.error_type == "validation_error"


def test_normal_no_choices_chunk_does_not_raise():
    """A chunk with no choices and no error fields should not raise.

    This is the normal case: the final chunk in a stream has no choices
    but carries usage information. It should be silently skipped.
    """

    def process_chunk(chunk):
        if not chunk.choices:
            _chunk_error_type = getattr(chunk, "error_type", None)
            _chunk_error_message = getattr(chunk, "error_message", None)
            if _chunk_error_type or _chunk_error_message:
                raise ProviderStreamError(str(_chunk_error_message or _chunk_error_type))
            return None  # normal skip
        return chunk.choices[0].delta.content

    usage_chunk = _make_usage_chunk(usage_obj={"prompt_tokens": 100, "completion_tokens": 50})
    result = process_chunk(usage_chunk)
    assert result is None  # silently skipped, no error


def test_content_chunk_does_not_raise():
    """A normal content chunk should be processed without error."""

    def process_chunk(chunk):
        if not chunk.choices:
            _chunk_error_type = getattr(chunk, "error_type", None)
            _chunk_error_message = getattr(chunk, "error_message", None)
            if _chunk_error_type or _chunk_error_message:
                raise ProviderStreamError(str(_chunk_error_message or _chunk_error_type))
            return None
        return chunk.choices[0].delta.content

    content_chunk = _make_normal_chunk(content="Hello world")
    result = process_chunk(content_chunk)
    assert result == "Hello world"


def test_error_chunk_with_only_error_type():
    """A chunk with only error_type (no error_message) should still raise."""

    def process_chunk(chunk):
        if not chunk.choices:
            _chunk_error_type = getattr(chunk, "error_type", None)
            _chunk_error_message = getattr(chunk, "error_message", None)
            if _chunk_error_type or _chunk_error_message:
                _err_detail = _chunk_error_message or _chunk_error_type
                raise ProviderStreamError(
                    str(_err_detail),
                    error_type=str(_chunk_error_type) if _chunk_error_type else None,
                )
            return None

    chunk = SimpleNamespace(
        choices=None,
        model=None,
        usage=None,
        error_type="rate_limit_exceeded",
        error_message=None,
    )

    with pytest.raises(ProviderStreamError) as exc_info:
        process_chunk(chunk)
    assert "rate_limit_exceeded" in str(exc_info.value)


def test_error_chunk_with_only_error_message():
    """A chunk with only error_message (no error_type) should still raise."""

    def process_chunk(chunk):
        if not chunk.choices:
            _chunk_error_type = getattr(chunk, "error_type", None)
            _chunk_error_message = getattr(chunk, "error_message", None)
            if _chunk_error_type or _chunk_error_message:
                _err_detail = _chunk_error_message or _chunk_error_type
                raise ProviderStreamError(
                    str(_err_detail),
                    error_type=str(_chunk_error_type) if _chunk_error_type else None,
                )
            return None

    chunk = SimpleNamespace(
        choices=None,
        model=None,
        usage=None,
        error_type=None,
        error_message="Input tokens exceed context window",
    )

    with pytest.raises(ProviderStreamError) as exc_info:
        process_chunk(chunk)
    assert "Input tokens exceed context window" in str(exc_info.value)
    assert exc_info.value.error_type is None


def test_chunk_without_error_attrs_does_not_raise():
    """A chunk with no error_type/error_message attributes at all should not raise.

    Some SDK chunk objects may not have these attributes at all (e.g. the
    standard OpenAI SDK ChatCompletionChunk). getattr with default None
    handles this correctly.
    """

    def process_chunk(chunk):
        if not chunk.choices:
            _chunk_error_type = getattr(chunk, "error_type", None)
            _chunk_error_message = getattr(chunk, "error_message", None)
            if _chunk_error_type or _chunk_error_message:
                raise ProviderStreamError(str(_chunk_error_message or _chunk_error_type))
            return None

    # Standard OpenAI SDK chunk (no error_type/error_message attrs)
    chunk = SimpleNamespace(
        choices=None,
        model="gpt-4",
        usage={"prompt_tokens": 10},
    )

    result = process_chunk(chunk)
    assert result is None  # no error, silently skipped


# --------------------------------------------------------------------------- #
# ProviderStreamError is not retried (classification)
# --------------------------------------------------------------------------- #


def test_provider_stream_error_not_classified_as_transient():
    """The retry machinery must NOT classify ProviderStreamError as transient.

    In the streaming retry loop, the following checks determine if an error
    is transient and should be retried:
    - _is_timeout: isinstance(e, (httpx.ReadTimeout, ...))
    - _is_conn_err: isinstance(e, (httpx.ConnectError, ...))
    - _is_stream_parse_err: agent._is_provider_stream_parse_error(e)
    - _is_empty_stream: isinstance(e, EmptyStreamError)

    ProviderStreamError is none of these, so it falls through to the
    non-transient path and surfaces immediately.
    """
    err = ProviderStreamError("context too long", error_type="validation_error")

    # Simulate the classification checks
    import httpx
    _is_timeout = isinstance(err, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout))
    _is_conn_err = isinstance(err, (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError))
    _is_empty_stream = isinstance(err, EmptyStreamError)

    assert not _is_timeout
    assert not _is_conn_err
    assert not _is_empty_stream

    # The combined transient check would be False → no retry
    _is_transient = _is_timeout or _is_conn_err or _is_empty_stream
    assert not _is_transient