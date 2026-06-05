"""Regression tests for streaming-unsupported error detection."""

from openai import APIError

import pytest

from agent.chat_completion_helpers import is_stream_unsupported_error


@pytest.mark.parametrize(
    "message",
    [
        "stream is not supported for this model",
        "Streaming is not supported for this model.",
        "streaming is not available on this endpoint",
        "Streaming is not enabled for the requested model",
        "This model does not support streaming.",
        "the selected model doesn't support stream mode",
        "stream: unsupported parameter",
        "unsupported request: stream is not allowed -> error code stream unsupported",
        "model cannot stream responses",
        "streaming is disabled for this deployment",
        "ERROR 400: STREAMING IS NOT SUPPORTED",
        "Bad request (param=stream): streaming is not supported here",
    ],
)
def test_detects_stream_unsupported_phrasings(message):
    assert is_stream_unsupported_error(Exception(message)) is True


def test_detects_on_real_openai_apierror():
    err = APIError(
        "This model does not support streaming.",
        request=None,  # type: ignore[arg-type]
        body=None,
    )
    assert is_stream_unsupported_error(err) is True


@pytest.mark.parametrize(
    "message",
    [
        "",
        "The requested model is not supported.",
        "model_not_supported",
        "HTTP 404: 404 page not found",
        "HTTP 400: invalid api key",
        "rate limit exceeded",
        "context length exceeded",
        "connection reset by peer",
        "The downstream service is not supported in your region",
        "downstream provider not supported",
        "upstream is not supported by the gateway",
        "this tool is not supported",
        "live streaming feed unavailable",
    ],
)
def test_ignores_unrelated_errors(message):
    assert is_stream_unsupported_error(Exception(message)) is False


def test_model_not_supported_does_not_downgrade_streaming():
    err = Exception("The requested model is not supported.")
    assert is_stream_unsupported_error(err) is False
