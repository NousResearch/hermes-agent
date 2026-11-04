"""Request-body byte caps reported as HTTP 400 (or status-less) route to payload_too_large."""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code, self.body = status_code, body or {}


# Verbatim bodies: a byte-capped gateway, Novita (20 MB) and Z.AI (code 1214).
_NOVITA = {"code": 400, "reason": "INVALID_REQUEST_BODY", "message": "request body too large, must be less than 20MB"}
_ZAI = {"code": 1214, "msg": "Request size exceeds the limit of {0}", "success": False}


@pytest.mark.parametrize("message,status,body", [
    ("Server received a request which exceeds maximum allowed content length. "
     "RequestSize(bytes): 34021227, Limit(bytes): 33554432.", 400, None),
    ("Request body size exceeds the maximum allowed limit", 400, None),
    ("Content-Length exceeds the maximum allowed request size", 400, None),
    ("Error code: 400 - " + str(_NOVITA), 400, _NOVITA),
    ("Error code: 400 - " + str(_ZAI), 400, _ZAI),
    ("Server received a request which exceeds maximum allowed content length. "
     "RequestSize(bytes): 34021227, Limit(bytes): 33554432.", None, None),
])
def test_request_body_byte_cap_compresses(message, status, body):
    result = classify_api_error(_APIError(message, status_code=status, body=body))
    assert (result.reason, result.retryable, result.should_compress) == (FailoverReason.payload_too_large, True, True)


@pytest.mark.parametrize("message", [
    "metadata length exceeds maximum allowed",
    "Invalid request body: temperature exceeds maximum allowed value",
    "tools[0].description content length exceeds maximum allowed",
    "response body size exceeds the maximum allowed",
    "request size parameter page_size exceeds the maximum of 100",
    # Token cap in byte wording (GitHub Models / Azure AI Inference, code tokens_limit_reached).
    "Request body too large for gpt-4.1 model. Max size: 8000 tokens.",
])
def test_field_or_response_length_errors_are_not_a_body_byte_cap(message):
    result = classify_api_error(_APIError(message, status_code=400))
    assert result.reason not in (FailoverReason.payload_too_large, FailoverReason.context_overflow), message
