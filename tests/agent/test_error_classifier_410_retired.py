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