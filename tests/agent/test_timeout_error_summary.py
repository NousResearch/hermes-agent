"""Transport timeouts get a phase-specific summary instead of the SDK's bare
"Request timed out." / "timed out" text (agent/timeout_error_summary.py)."""

import httpx
import openai

from agent.api_error_summary import ApiErrorSummaryMixin

_REQ = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")


def _sdk_timeout(cause: Exception) -> openai.APITimeoutError:
    # Same shape the openai SDK raises: ``raise APITimeoutError(request=...) from err``.
    try:
        raise openai.APITimeoutError(request=_REQ) from cause
    except openai.APITimeoutError as exc:
        return exc


def test_sdk_wrapped_timeouts_name_the_phase():
    summarize = ApiErrorSummaryMixin._summarize_api_error
    connect = summarize(_sdk_timeout(httpx.ConnectTimeout("timed out", request=_REQ)))
    read = summarize(_sdk_timeout(httpx.ReadTimeout("timed out", request=_REQ)))
    midstream = summarize(httpx.ReadTimeout("timed out", request=_REQ))

    assert connect.startswith("Connection to the provider timed out before it was established")
    for text in (read, midstream):
        assert text.startswith("Request timed out waiting for the provider")
        assert "request_timeout_seconds" in text


def test_non_timeout_errors_keep_their_summary():
    assert ApiErrorSummaryMixin._summarize_api_error(RuntimeError("upstream exploded")) == "upstream exploded"
