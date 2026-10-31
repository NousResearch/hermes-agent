"""Regression: empty-body HTTP 4xx errors must still surface a real provider message.

Reported on Windows (#36109): an LLM API call returned HTTP 400 with an *empty*
parsed SDK ``body`` ({}), so ``_summarize_api_error`` fell through to the bare
``str(error)`` path and the user saw only "HTTP 400" with no provider detail.
The SDK leaves ``body`` empty in this case, but the underlying httpx
``response`` still carries the real payload in ``.text``. These tests lock the
contract: when ``body`` is empty, fall back to ``response.text`` (parsing a JSON
``error.message`` / ``message`` when present) so logs and CLI show the real
provider error. This is a diagnostic improvement and is platform-agnostic.
"""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from run_agent import AIAgent


def _make_empty_body_error(response_text: str, status_code: int = 400) -> Exception:
    """Mimic an OpenAI-SDK error whose parsed body is empty but whose httpx
    response still holds the payload text."""
    err = Exception("")  # str(error) is empty/uninformative on this path
    err.status_code = status_code
    err.body = {}  # empty dict — the #36109 trigger
    err.response = SimpleNamespace(text=response_text)
    return err


def test_empty_body_falls_back_to_response_json_error_message():
    """A JSON payload with error.message is surfaced (not a bare HTTP 400)."""
    err = _make_empty_body_error(
        '{"error": {"message": "model `foo` does not exist", "type": "invalid_request_error"}}'
    )
    summary = AIAgent._summarize_api_error(err)
    assert "HTTP 400" in summary
    assert "model `foo` does not exist" in summary

@pytest.mark.parametrize(
    "technical_message",
    [
        "Temporary failure in name resolution",
        "Name or service not known",
        "nodename nor servname provided, or not known",
        "getaddrinfo failed",
        "No address associated with hostname",
        "Network is unreachable",
    ],
)
def test_network_resolution_failure_explains_that_the_user_may_be_offline(
    technical_message,
):
    error = OSError(-3, technical_message)

    summary = AIAgent._summarize_api_error(error)

    assert "You may be offline" in summary
    assert "name resolution" not in summary.lower()


def test_wrapped_dns_resolution_failure_gets_the_same_friendly_message():
    try:
        try:
            raise OSError(-3, "Temporary failure in name resolution")
        except OSError as cause:
            raise RuntimeError("Connection error.") from cause
    except RuntimeError as error:
        summary = AIAgent._summarize_api_error(error)

    assert "You may be offline" in summary
    assert "Connection error" not in summary


def test_unread_streaming_response_does_not_crash_and_falls_back_to_exception_message():
    """Unread streaming responses must not replace the real provider error."""

    class _StreamingError(Exception):
        def __init__(self):
            super().__init__("Gemini HTTP 429: quota exceeded")
            self.status_code = 429
            self.response: Any = None

    err = _StreamingError()

    class _UnreadStreamingResponse:
        @property
        def text(self):
            raise httpx.ResponseNotRead()

    err.response = _UnreadStreamingResponse()
    summary = AIAgent._summarize_api_error(err)
    assert "HTTP 429" in summary
    assert "Gemini HTTP 429: quota exceeded" in summary



def test_provider_request_id_reaches_failure_line_but_not_summary():
    """A rejected call names the provider's request id (OpenAI ``x-request-id``, Anthropic
    ``request-id``) on the user-facing failure line so a support ticket can cite it, while the
    result's ``error`` summary stays id-free: cron incident signing hashes it, and a per-request id
    would mint a new incident per failure."""
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import nonretryable_client_error_result

    class _Agent:
        log_prefix, verbose = "", False
        _summarize_api_error = staticmethod(AIAgent._summarize_api_error)

        def __init__(self):
            self.status_lines = []

        def _emit_diagnostic_status(self, line):
            self.status_lines.append(line)

        def __getattr__(self, name):  # persist/print helpers the terminal path also calls
            return lambda *args, **kwargs: None

    for header in ("x-request-id", "request-id"):
        body = {"error": {"message": "Incorrect API key provided"}}
        response = httpx.Response(
            401, headers={header: "req_abc123"}, json=body,
            request=httpx.Request("POST", "https://api.example.test/v1/chat/completions"),
        )
        err = Exception("HTTP 401: Incorrect API key provided")
        err.status_code, err.response, err.body = 401, response, body
        wrapped = RuntimeError("retry exhausted")
        wrapped.__cause__ = err  # the id is found through the exception chain too
        for error in (err, wrapped):
            agent = _Agent()
            result = nonretryable_client_error_result(
                agent, error, classify_api_error(err, provider="custom", model="m"), status_code=401,
                api_kwargs=None, api_messages=[], messages=[], conversation_history=None, api_call_count=1,
                approx_tokens=10, provider="custom", base_url="https://api.example.test/v1", model="m",
            )
            assert any(line.endswith("(provider request id: req_abc123)") for line in agent.status_lines)
            assert "req_abc123" not in result["error"]
