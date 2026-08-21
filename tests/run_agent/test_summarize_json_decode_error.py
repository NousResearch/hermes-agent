"""Regression: json.JSONDecodeError from empty provider responses must be
summarized into a readable one-liner.

When an upstream (or a proxy in front of it) returns an HTTP 200 with an
empty — or otherwise non-JSON — body, the OpenAI/Anthropic SDK raises
``json.JSONDecodeError`` ("Expecting value: line 1 column 1 (char 0)").
Before the fix, ``_summarize_api_error`` fell through to the raw
``str(error)`` fallback, so the cryptic message reached cron delivery
with no indication the provider channel was dead.
"""
import json

from run_agent import AIAgent


def _make_empty_response_error() -> json.JSONDecodeError:
    """Simulate the SDK parsing an empty 200 response body."""
    err = json.JSONDecodeError("Expecting value: line 1 column 1 (char 0)", "", 0)
    err.status_code = 200
    return err


def test_summarize_json_decode_error_is_readable():
    """``_summarize_api_error`` must turn a bare JSONDecodeError into a
    readable one-liner about an empty/malformed upstream response."""
    summary = AIAgent._summarize_api_error(_make_empty_response_error())

    # The raw SDK message must not leak through.
    assert "Expecting value" not in summary.lower()

    # The summary should mention the HTTP status so the user can see it's
    # a server-side issue, not a local config error.
    assert "200" in summary

    # A one-liner — not a multi-line traceback.
    assert len(summary) < 200

    # Must mention the actual problem (empty/malformed response).
    assert "malformed" in summary.lower() or "empty" in summary.lower()


def test_summarize_json_decode_error_no_status_code():
    """Handle JSONDecodeError without a status_code attribute gracefully."""
    err = json.JSONDecodeError("Expecting value: line 1 column 1 (char 0)", "", 0)
    # Do NOT set status_code — some SDK paths don't attach it.
    summary = AIAgent._summarize_api_error(err)

    assert "Expecting value" not in summary.lower()
    assert "malformed" in summary.lower() or "empty" in summary.lower()
    assert len(summary) < 200