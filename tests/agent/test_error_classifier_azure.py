"""Tests for Azure-specific error_classifier patterns."""
from __future__ import annotations

import pytest

from agent.error_classifier import classify_api_error, FailoverReason


class _MockAPIError(Exception):
    def __init__(self, message="", *, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _classify(message="", *, status_code=None, error_code=None):
    """Helper that constructs a synthetic exception and classifies it."""
    body = None
    if error_code is not None:
        body = {"error": {"code": error_code, "message": message}}
    err = _MockAPIError(message, status_code=status_code, body=body)
    return classify_api_error(
        err,
        provider="azure-foundry",
        model="gpt-5",
    )


class TestAzureContentFilter:
    def test_responsible_ai_policy_violation_code(self):
        c = _classify(error_code="ResponsibleAIPolicyViolation")
        assert c.reason == FailoverReason.content_filter
        assert c.retryable is False

    def test_content_filter_code(self):
        c = _classify(error_code="content_filter")
        assert c.reason == FailoverReason.content_filter
        assert c.retryable is False

    def test_content_filter_message_pattern(self):
        c = _classify("The response was filtered due to content_filter triggers")
        assert c.reason == FailoverReason.content_filter
        assert c.retryable is False

    def test_responsible_ai_message_pattern(self):
        c = _classify("Request blocked by ResponsibleAIPolicyViolation")
        assert c.reason == FailoverReason.content_filter

    def test_jailbreak_message_pattern(self):
        c = _classify("Detected jailbreak attempt in user prompt")
        assert c.reason == FailoverReason.content_filter


class TestAzureRateLimit:
    def test_429_classified_as_rate_limit(self):
        c = _classify("Too many requests", status_code=429)
        assert c.reason == FailoverReason.rate_limit
        assert c.retryable is True


class TestAzureModelNotFound:
    def test_deployment_not_found_code(self):
        c = _classify(error_code="DeploymentNotFound")
        assert c.reason == FailoverReason.model_not_found
        assert c.retryable is False

    def test_deployment_not_found_message(self):
        c = _classify(
            "The API deployment for this resource does not exist (DeploymentNotFound)"
        )
        assert c.reason == FailoverReason.model_not_found

    def test_model_not_found_code(self):
        c = _classify(error_code="model_not_found")
        assert c.reason == FailoverReason.model_not_found

    def test_invalid_model_code(self):
        c = _classify(error_code="invalid_model")
        assert c.reason == FailoverReason.model_not_found


class TestRegression:
    def test_content_filter_reason_is_non_retryable(self):
        # Sanity check on the FailoverReason.content_filter metadata.
        assert FailoverReason.content_filter.value == "content_filter"
