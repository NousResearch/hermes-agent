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


def test_pool_and_write_timeouts_get_no_provider_response_hint():
    for cause in (httpx.PoolTimeout("timed out"), httpx.WriteTimeout("timed out", request=_REQ)):
        assert ApiErrorSummaryMixin._summarize_api_error(_sdk_timeout(cause)) == "Request timed out."


def test_error_raised_while_handling_a_timeout_keeps_its_own_summary():
    # __context__ is not a cause: a 401 raised inside ``except ReadTimeout`` is an auth failure.
    resp = httpx.Response(401, request=_REQ, json={"error": {"message": "Invalid API key"}})
    try:
        try:
            raise httpx.ReadTimeout("timed out", request=_REQ)
        except httpx.ReadTimeout:
            raise openai.AuthenticationError("Error code: 401", response=resp, body={"error": {"message": "Invalid API key"}})
    except openai.AuthenticationError as exc:
        assert ApiErrorSummaryMixin._summarize_api_error(exc) == "HTTP 401: Invalid API key"


def test_read_timeout_hint_names_the_agents_provider_key():
    class _Agent(ApiErrorSummaryMixin):
        provider = "openrouter"

    error = _sdk_timeout(httpx.ReadTimeout("timed out", request=_REQ))
    assert "providers.openrouter.request_timeout_seconds" in _Agent()._summarize_api_error(error)
    assert "providers.<provider>.request_timeout_seconds" in ApiErrorSummaryMixin._summarize_api_error(error)
