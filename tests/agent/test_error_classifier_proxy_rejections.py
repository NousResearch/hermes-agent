"""Descriptive upstream rejections must not trigger the large-session overflow guess."""

import json

import httpx
import pytest
from openai import APIStatusError

from agent.error_classifier import FailoverReason, classify_api_error


def _error(body):
    request = httpx.Request("POST", "https://proxy.example.test/v1/chat/completions")
    return APIStatusError("Error", response=httpx.Response(400, request=request), body=body)


@pytest.mark.parametrize("field", ["errorMessage", "responseBody", "reason"])
@pytest.mark.parametrize("message, expected", [
    ("Unsupported parameter: 'response_format' on this endpoint", FailoverReason.format_error),
    ("The upstream request policy has rejected this request", FailoverReason.format_error),
    ("Input exceeds the maximum context length of this model", FailoverReason.context_overflow),
])
def test_nested_proxy_reason_drives_classification_instead_of_outer_error(field, message, expected):
    body = {"message": "Error", "parameters": {
        field: "Optional[" + json.dumps({"error": {"message": message}}) + "]",
    }}
    result = classify_api_error(_error(body), approx_tokens=100000, num_messages=100)
    assert result.reason is expected
    assert result.should_compress is (expected is FailoverReason.context_overflow)
    # Classification reads nested details; the public message still preserves the SDK error.
    assert result.message == "Error"


@pytest.mark.parametrize("body, expected", [
    ({"message": "Error"}, FailoverReason.context_overflow),
    ({"message": "Error", "error": {"message": "The upstream request policy has rejected this request"}},
     FailoverReason.format_error),
    ({"error": {"message": "Input exceeds the maximum context length of this model"}},
     FailoverReason.context_overflow),
    ({"message": "The upstream request policy has rejected this request"}, FailoverReason.format_error),
])
def test_existing_descriptive_and_bare_400_recovery_are_preserved(body, expected):
    result = classify_api_error(_error(body), approx_tokens=100000, num_messages=100)
    assert result.reason is expected
    assert result.should_compress is (expected is FailoverReason.context_overflow)
