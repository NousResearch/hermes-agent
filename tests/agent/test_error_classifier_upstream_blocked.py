"""A 403 written by a WAF/CDN in front of the provider is not an API-key rejection (#53099, #70566).

A relay that blocks the SDK User-Agent answers ``403 Your request was blocked.``; Cloudflare's
browser challenge answers 403 HTML. Both used to classify as ``auth`` and print key guidance.
"""
import json

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _Response:
    def __init__(self, *, headers=None, text=""):
        self.headers = headers or {}
        self.text = text

    def json(self):
        raise ValueError("not JSON")


class _APIError(Exception):
    def __init__(self, message, status_code, *, body=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response


@pytest.mark.parametrize("body", [
    "Error code: 403 - Your request was blocked.",
    "<!doctype html><html><body>Enable JavaScript and cookies to continue</body></html>",
    "<!doctype html><html><script src='/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page'></script></html>",
])
def test_403_waf_block_is_upstream_blocked_not_auth(body):
    result = classify_api_error(_APIError(body, 403), provider="openai-api")
    assert result.reason == FailoverReason.upstream_blocked
    assert result.retryable is False and result.should_fallback is True
    assert result.should_rotate_credential is False and result.is_auth is False


@pytest.mark.parametrize("error", [
    _APIError(
        "Error code: 403 - Forbidden",
        403,
        response=_Response(headers={"cf-mitigated": "challenge"}),
    ),
    _APIError(
        "Error code: 403 - Forbidden",
        403,
        body="<!doctype html><html>Enable JavaScript and cookies to continue</html>",
    ),
    _APIError(
        "Error code: 403 - Forbidden",
        403,
        response=_Response(
            text="<!doctype html><script src='/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page'></script>"
        ),
    ),
])
def test_403_waf_evidence_outside_exception_message_is_upstream_blocked(error):
    result = classify_api_error(error, provider="openai-codex")
    assert result.reason == FailoverReason.upstream_blocked
    assert result.should_rotate_credential is False and result.is_auth is False


@pytest.mark.parametrize("error", [
    _APIError(
        "Error code: 403 - Incorrect API key provided",
        403,
        body={"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}},
        response=_Response(
            headers={"content-type": "application/json"},
            text='{"metadata":"cf-browser-verification"}',
        ),
    ),
    _APIError(
        "Error code: 401 - Unauthorized",
        401,
        body="<!doctype html><html>Enable JavaScript and cookies to continue</html>",
        response=_Response(headers={"cf-mitigated": "challenge"}),
    ),
])
def test_auth_responses_are_not_overridden_by_unrelated_waf_evidence(error):
    assert classify_api_error(error, provider="openai-codex").reason == FailoverReason.auth


def test_real_sdk_structured_auth_metadata_stays_auth():
    body = {
        "error": {"message": "Incorrect API key provided", "code": "invalid_api_key"},
        "metadata": {"diagnostic_label": "cf-browser-verification"},
    }
    response = httpx.Response(
        403,
        json=body,
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )
    error = openai.PermissionDeniedError(
        f"Error code: 403 - {body!r}", response=response, body=body,
    )

    assert classify_api_error(error, provider="openai-codex").reason == FailoverReason.auth


@pytest.mark.parametrize("as_bytes", [False, True])
def test_oversized_json_auth_body_stays_auth(as_bytes):
    payload = json.dumps({
        "error": {"message": "Incorrect API key provided", "code": "invalid_api_key"},
        "metadata": "cf-browser-verification",
        "padding": "x" * 70000,
    })
    body = payload.encode() if as_bytes else payload
    error = _APIError(f"Error code: 403 - {payload}", 403, body=body)

    assert classify_api_error(error, provider="openai-codex").reason == FailoverReason.auth


@pytest.mark.parametrize("body, status, reason", [
    ("<html><title>Forbidden</title><body>Access denied</body></html>", 403, FailoverReason.auth),
    ("<html>Enable JavaScript and cookies to continue</html>", 401, FailoverReason.auth),
])
def test_generic_403_and_all_401_keep_auth(body, status, reason):
    assert classify_api_error(_APIError(body, status), provider="openai-api").reason == reason
