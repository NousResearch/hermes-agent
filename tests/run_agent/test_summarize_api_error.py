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


def test_empty_body_falls_back_to_raw_response_text_when_not_json():
    """A non-JSON response body is surfaced verbatim (truncated), not dropped."""
    err = _make_empty_body_error("upstream connect error or disconnect/reset before headers")
    summary = AIAgent._summarize_api_error(err)
    assert "HTTP 400" in summary
    assert "upstream connect error" in summary


def test_empty_body_fallback_redacts_secrets(monkeypatch):
    """The surfaced provider/proxy error body must pass through the secret
    redactor — a proxy echoing an API key in the error must not leak it into
    final_response/logs (the empty-body path previously hid it as bare HTTP 400)."""
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    err = _make_empty_body_error(
        '{"error": {"message": "bad key: sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef"}}'
    )
    summary = AIAgent._summarize_api_error(err)
    assert "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef" not in summary

