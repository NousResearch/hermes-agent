"""A 403 written by a WAF/CDN in front of the provider is not an API-key rejection (#53099, #70566).

A relay that blocks the SDK User-Agent answers ``403 Your request was blocked.``; Cloudflare's
browser challenge answers 403 HTML. Both used to classify as ``auth`` and print key guidance.
"""
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


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


@pytest.mark.parametrize("body, status, reason", [
    ("<html><title>Forbidden</title><body>Access denied</body></html>", 403, FailoverReason.auth),
    ("<html>Enable JavaScript and cookies to continue</html>", 401, FailoverReason.auth),
])
def test_generic_403_and_all_401_keep_auth(body, status, reason):
    assert classify_api_error(_APIError(body, status), provider="openai-api").reason == reason


@pytest.mark.parametrize("body", [
    "Error code: 403 - Request blocked: prompt injection patterns detected",
    "Error code: 403 - Request blocked: prompt injection patterns detected "
    "metadata: {'patterns': ['role_tag_injection']}",
    "Error code: 403 - Request blocked metadata: {'patterns': ['role_delimiter_injection']}",
])
def test_openrouter_guardrail_403_is_content_policy_not_waf(body):
    """OpenRouter's injection guardrail rejects the request CONTENT before any model sees
    it (#132504): it shares the "request blocked" wording with a WAF block, but firewall /
    User-Agent advice cannot fix request content — and the session keeps 403ing until the
    offending token leaves the context, so it must not read as a credential problem either."""
    result = classify_api_error(_APIError(body, 403), provider="openrouter")
    assert result.reason == FailoverReason.content_policy_blocked
    assert result.retryable is False and result.should_fallback is True
    assert result.should_rotate_credential is False and result.is_auth is False
