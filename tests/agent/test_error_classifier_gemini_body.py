"""Gemini error bodies must survive consumed streaming responses for classification."""

import json

import httpx

from agent.error_classifier import FailoverReason, classify_api_error
from agent.gemini_native_adapter import gemini_http_error


ORDER_MESSAGE = (
    "Please ensure that function call turn comes immediately after a user turn "
    "or after a function response turn."
)
QUOTA_MESSAGE = (
    "You exceeded your current quota. limit: 0, metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests"
)


def _error(message, *, status=400, json_response=False):
    payload = {"error": {"code": status, "status": "RESOURCE_EXHAUSTED" if status == 429 else "INVALID_ARGUMENT", "message": message}}
    return _from_body(json.dumps(payload), status=status, payload=payload if json_response else None)


def _from_body(body_text, *, status=400, payload=None):
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-neon:streamGenerateContent")
    resp = httpx.Response(status, request=req) if payload is None else httpx.Response(status, json=payload, request=req)
    return gemini_http_error(
        resp, body_text=body_text, api_key="AIzaDummyKey",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )


def _classify_large(err):
    return classify_api_error(
        err, provider="gemini", model="gemini-4-neon",
        approx_tokens=584_179, context_length=1_048_576, num_messages=395,
    )


def test_gemini_http_error_carries_json_error_body():
    err = _error(ORDER_MESSAGE)
    assert hasattr(err, "body")
    assert isinstance(err.body, dict)
    assert err.body["error"]["status"] == "INVALID_ARGUMENT"
    assert err.body["error"]["code"] == 400
    assert err.body["error"]["message"] == str(err)
    assert err.details["message"] == ORDER_MESSAGE


def test_gemini_descriptive_400_on_large_session_is_not_context_overflow():
    classified = _classify_large(_error(ORDER_MESSAGE))
    assert classified.reason == FailoverReason.format_error
    assert classified.should_compress is False
    assert classified.should_fallback is True


def test_gemini_short_json_message_on_large_session_is_format_error():
    classified = _classify_large(_error("Error"))
    assert classified.reason == FailoverReason.format_error


def test_gemini_json_response_path_message_is_unified():
    err = _error(QUOTA_MESSAGE, status=429, json_response=True)
    classified = classify_api_error(err, provider="gemini", model="gemini-4-neon")
    assert classified.message == str(err)[:500]
    assert "free tier" in classified.message.lower()


def test_gemini_classified_message_keeps_guidance():
    err = _error(QUOTA_MESSAGE, status=429)
    classified = classify_api_error(err, provider="gemini", model="gemini-4-neon")
    assert classified.reason == FailoverReason.rate_limit
    assert classified.message == str(err)[:500]
    assert "free tier" in str(err).lower()


def test_gemini_non_json_body_leaves_body_none_and_heuristic_intact():
    err = _from_body("Error")
    assert getattr(err, "body", None) is None
    assert _classify_large(err).reason == FailoverReason.context_overflow


def test_gemini_empty_error_object_leaves_body_none():
    for payload in ({"error": {}}, {"error": "x"}, {"message": "no error key"}):
        err = _from_body(json.dumps(payload))
        assert getattr(err, "body", None) is None


def test_gemini_true_overflow_message_still_context_overflow():
    err = _error("The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).")
    classified = _classify_large(err)
    assert classified.reason == FailoverReason.context_overflow
    assert classified.should_compress is True
