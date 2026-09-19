"""upstream_error structured payloads classify as server_error with fallback.

Regression test for #55096: structured ``upstream_error`` payloads (the code
``_code_from_payload`` pulls from ``error.type``) fell through ``_ERROR_CODE_VERDICTS``
to ``unknown`` and missed fallback-worthiness.
"""

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


def _upstream_error_exc():
    # Anthropic's structured upstream_error shape: the provider's own outage,
    # surfaced via error.type with no status code the classifier can route on.
    return MockAPIError(
        "provider error",
        body={"error": {"type": "upstream_error", "message": "The provider is not responding"}},
    )


def test_upstream_error_classifies_as_server_error():
    verdict = classify_api_error(_upstream_error_exc(), provider="anthropic", model="m")
    assert verdict.reason == FailoverReason.server_error


def test_upstream_error_is_fallback_worthy():
    # Nothing is wrong with the key — the upstream is down — so route to the
    # next provider/model instead of retrying the same one.
    verdict = classify_api_error(_upstream_error_exc(), provider="anthropic", model="m")
    assert verdict.should_fallback is True
