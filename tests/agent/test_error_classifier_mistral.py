"""Mistral's field-disabled 400 uses the drop-reasoning rung (#119249)."""

import httpx
from openai import BadRequestError

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    is_reasoning_field_rejection,
)


def test_mistral_reasoning_effort_not_enabled_is_reasoning_mandatory():
    for message, rejected in (
        ("reasoning_effort is not enabled for this model", True),
        ("Function calling is not enabled for this model", False),
        ("Invalid model: codestral-2508", False),
    ):
        assert is_reasoning_field_rejection(message) is rejected
        error = BadRequestError(
            message,
            response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/v1")),
            body=None,
        )
        result = classify_api_error(error, provider="custom", model="codestral-2508")
        assert (result.reason == FailoverReason.reasoning_mandatory) is rejected
        if rejected:
            assert result.retryable is True and result.should_fallback is False
