"""Prompt-level Gemini refusal contracts; regression for #121317."""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.gemini_native_adapter import GeminiAPIError, translate_gemini_response, translate_stream_event


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["SAFETY", "OTHER", "FUTURE_BLOCK_REASON"])
def test_prompt_block_is_structured_terminal_error_without_synthetic_http_status(stream, reason):
    payload = {"promptFeedback": {"blockReason": reason},
               "usageMetadata": {"promptTokenCount": 900, "totalTokenCount": 900}}
    with pytest.raises(GeminiAPIError, match=reason) as caught:
        if stream:
            translate_stream_event(payload, "gemini-3-flash", {})
        else:
            translate_gemini_response(payload, "gemini-3-flash")
    error = caught.value
    assert error.code == "gemini_prompt_blocked"
    assert error.status_code is None
    assert error.details["promptFeedback"] == payload["promptFeedback"]
    assert error.usage.prompt_tokens == error.usage.total_tokens == 900
    assert error.usage.completion_tokens == 0
    verdict = classify_api_error(error, provider="gemini")
    assert verdict.reason == FailoverReason.content_policy_blocked
    assert verdict.retryable is False
    assert verdict.should_fallback is True
    assert verdict.status_code is None


@pytest.mark.parametrize("feedback", [None, {}, {"safetyRatings": []}, {"blockReason": None},
    {"blockReason": ""}, {"blockReason": "  "}, {"blockReason": 0}, {"blockReason": True},
    {"blockReason": "BLOCK_REASON_UNSPECIFIED"}, {"blockReason": " block_reason_unspecified "}])
def test_empty_and_usage_only_events_do_not_become_prompt_blocks(feedback):
    payload = {"promptFeedback": feedback, "usageMetadata": {"promptTokenCount": 5}}
    assert translate_stream_event(payload, "gemini-3-flash", {}) == []
    response = translate_gemini_response(payload, "gemini-3-flash")
    assert response.choices[0].message.content == ""
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("reason,finish", [("STOP", "stop"), ("SAFETY", "content_filter"),
    ("RECITATION", "content_filter"), ("PROHIBITED_CONTENT", "stop")])
def test_candidates_keep_existing_finish_and_usage_despite_prompt_feedback(reason, finish):
    payload = {"promptFeedback": {"blockReason": "SAFETY"},
               "candidates": [{"content": {"parts": []}, "finishReason": reason}],
               "usageMetadata": {"promptTokenCount": 900, "totalTokenCount": 900}}
    response = translate_gemini_response(payload, "gemini-3-flash")
    chunks = translate_stream_event(payload, "gemini-3-flash", {})
    assert response.choices[0].finish_reason == finish
    assert chunks[-1].choices[0].finish_reason == finish
    assert response.usage.prompt_tokens == chunks[-1].usage.prompt_tokens == 900


@pytest.mark.parametrize("provider", ["gemini", "google", ""])
def test_classifier_uses_structured_code_without_message_matching(provider):
    error = GeminiAPIError("provider refused request", code="gemini_prompt_blocked")
    verdict = classify_api_error(error, provider=provider)
    assert verdict.reason == FailoverReason.content_policy_blocked
    assert verdict.retryable is False


def test_classifier_keeps_body_code_precedence_over_exception_code():
    error = GeminiAPIError("request rejected", code="gemini_prompt_blocked")
    error.body = {"error": {"code": "context_length_exceeded"}}
    assert classify_api_error(error, provider="gemini").reason == FailoverReason.context_overflow


@pytest.mark.parametrize("code,reason", [("context_length_exceeded", FailoverReason.context_overflow),
    ("future_unknown_code", FailoverReason.unknown), (429, FailoverReason.unknown)])
def test_exception_code_fallback_preserves_other_provider_classification(code, reason):
    error = Exception("opaque provider failure")
    error.code = code
    assert classify_api_error(error, provider="custom").reason == reason
