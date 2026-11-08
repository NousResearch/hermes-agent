"""Revoked OAuth tokens in HTTP-200 SSE errors use auth recovery (#139871)."""

import json

import anthropic
import httpx
import pytest

from agent.anthropic_adapter import normalize_stream_usage
from agent.error_classifier import FailoverReason, classify_api_error


@pytest.mark.parametrize("provider", ["custom", "anthropic"])
@pytest.mark.parametrize("message", [
    "[upstream] [401]: OAuth access token has been revoked.",
    "OAuth access token has been revoked.",
])
def test_revoked_token_stream_uses_auth_recovery(message, provider):
    payload = {"type": "error", "error": {"message": message, "retry_after": 7200}}
    frame = f"event: error\ndata: {json.dumps(payload)}\n\n".encode()
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=frame,
    ))
    with anthropic.Anthropic(
        api_key="dummy-key", base_url="https://example.invalid", max_retries=0,
        http_client=httpx.Client(transport=transport, trust_env=False),
    ) as client:
        with pytest.raises(anthropic.APIStatusError) as raised:
            with client.messages.stream(
                model="test-model", max_tokens=1,
                messages=[{"role": "user", "content": "hello"}],
            ) as stream:
                list(normalize_stream_usage(stream))

    assert raised.value.status_code == 200
    result = classify_api_error(raised.value, provider=provider)
    assert result.reason is FailoverReason.auth
    assert result.retryable is False
    assert result.should_fallback is True
    assert result.should_rotate_credential is True
    assert result.should_compress is False
    assert result.status_code == 200


@pytest.mark.parametrize("status,message,code,reason,retryable,fallback,rotate", [
    pytest.param(401, "Rate limit exceeded", None, "auth", False, True, True, id="http-401"),
    pytest.param(403, "Rate limit exceeded", None, "auth", False, True, False, id="http-403"),
    pytest.param(429, "OAuth access token has been revoked.", None,
                 "rate_limit", True, True, True, id="http-429"),
    pytest.param(500, "OAuth access token has been revoked.", None,
                 "server_error", True, False, False, id="http-500"),
    pytest.param(502, "OAuth access token has been revoked.", None,
                 "server_error", True, False, False, id="http-502"),
    pytest.param(503, "OAuth access token has been revoked.", None,
                 "overloaded", True, False, False, id="http-503"),
    pytest.param(529, "OAuth access token has been revoked.", None,
                 "overloaded", True, False, False, id="http-529"),
    pytest.param(599, "OAuth access token has been revoked.", None,
                 "server_error", True, False, False, id="http-other-5xx"),
    pytest.param(None, "Failure", 403, "auth", False, True, False, id="numeric-code-403"),
    pytest.param(None, "Failure", 429, "rate_limit", True, True, True, id="numeric-code-429"),
    pytest.param(None, "invalid_api_key", None, "auth", False, True, True, id="symbolic-auth"),
    pytest.param(None, "token revoked", None, "auth", False, True, True, id="existing-revocation"),
    pytest.param(None, "Failure", "rate_limit_error",
                 "rate_limit", True, True, True, id="symbolic-rate-limit"),
    pytest.param(200, "Too many requests: OAuth access token has been revoked.", None,
                 "rate_limit", True, True, True, id="rate-limit-message-precedence"),
    pytest.param(200, "[upstream] [401]: temporary glitch", None,
                 "unknown", True, False, False, id="no-status-from-prose"),
])
def test_stream_auth_preserves_existing_classification(
    status, message, code, reason, retryable, fallback, rotate,
):
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    body = {"error": {"message": message}}
    if code is not None:
        body["error"]["code"] = code
    if status is None:
        error = anthropic.APIError("Stream failed", request=request, body=body)
    else:
        error = anthropic.APIStatusError(
            "Request failed", response=httpx.Response(status, request=request), body=body,
        )

    result = classify_api_error(error, provider="anthropic")
    assert (result.reason.value, result.retryable, result.should_fallback,
            result.should_rotate_credential) == (reason, retryable, fallback, rotate)
    assert result.should_compress is False
    expected_status = status if status is not None else code if isinstance(code, int) else None
    assert result.status_code == expected_status
