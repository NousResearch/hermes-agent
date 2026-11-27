"""Tests for HTTP 410 model-retired classification in agent.error_classifier.

Split from test_error_classifier.py to keep the parent under the 2000-line
lint target. See PR #147812.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_error_classifier import MockAPIError


class TestClassify410Retired:
    """HTTP 410 Gone -- model permanently retired by the provider."""

    def test_410_retired_model_is_model_not_found(self):
        # HTTP 410 Gone: provider permanently retired the model slug.
        # Must fall back to a different model, not burn retries on a
        # format_error that will never succeed.
        e = MockAPIError(
            "Gone",
            status_code=410,
            body={
                "status": 410,
                "message": (
                    "deepseek-v4-flash:0731 was retired at 2026-09-25 00:00:00 "
                    "-0700 PDT. Please use a current model."
                ),
            },
        )
        result = classify_api_error(e, provider="nous", model="deepseek-v4-flash:0731")
        assert result.reason == FailoverReason.model_not_found
        assert result.retryable is False
        assert result.should_fallback is True

    def test_410_without_retired_message_stays_format_error(self):
        # A 410 that does not mention retirement stays format_error
        # (the pre-fix 4xx default) so unrelated 410s are not silently
        # misrouted to model_not_found.
        e = MockAPIError("Gone", status_code=410)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

@pytest.mark.parametrize("provider, body", [
    # Live bodies (2026-11-26) for a retired id: Gemini's 404 and xAI's 404.
    ("gemini", [{"error": {"code": 404, "status": "NOT_FOUND", "message": (
        "models/gemini-1.0-pro is not found for API version v1beta, or is not supported for "
        "generateContent. Call ModelService.ListModels to see the list of available models and their "
        "supported methods.")}}]),
    ("xai", {"code": "not-found", "error": (
        "The model grok-beta was deprecated on 2025-09-15 and is no longer accessible via the API. "
        "Please use grok-3 instead.")}),
])
def test_retired_model_404_falls_back_instead_of_retrying(provider, body):
    e = MockAPIError(f"Error code: 404 - {body}", status_code=404, body=body)
    result = classify_api_error(e, provider=provider, model="m")
    assert result.reason == FailoverReason.model_not_found
    assert result.retryable is False and result.should_fallback is True
