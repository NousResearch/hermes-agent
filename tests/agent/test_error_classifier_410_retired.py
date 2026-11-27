"""A model the provider retired or does not know: HTTP 410 and 404 wordings that must fall back.

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


def test_404_model_is_not_found_wording():
    """OpenAI-compatible gateways can word it as ``model is not found`` (the copula breaks
    the ``model not found`` substring): a deterministic failure, not a retry."""
    result = classify_api_error(MockAPIError("model is not found", status_code=404))
    assert result.reason == FailoverReason.model_not_found
    assert result.should_fallback is True and result.retryable is False


def _anthropic_404(message):
    body = {"type": "error", "error": {"type": "not_found_error", "message": message}}
    return MockAPIError("Error code: 404", status_code=404, body=body)


def test_404_echoing_the_requested_model_id_is_model_not_found():
    """Anthropic answers an unknown model with the id we sent (``model: <id>``)."""
    result = classify_api_error(_anthropic_404("model: some-model-id"), provider="anthropic", model="some-model-id")
    assert result.reason == FailoverReason.model_not_found
    assert result.retryable is False and result.should_fallback is True


@pytest.mark.parametrize("message", [
    *(f"model: {m}" for m in ("some-model-id-extra", "some-model-id.v2", "some-model-id/variant",
                               "some-model-id_2", "some-model-id+variant")),
    "Not found",  # a wrong endpoint path names no model: it must not undo #14013
])
def test_404_not_naming_the_requested_model_stays_generic(message):
    result = classify_api_error(_anthropic_404(message), provider="anthropic", model="some-model-id")
    assert result.reason == FailoverReason.unknown
    assert result.retryable is True and result.should_fallback is False
