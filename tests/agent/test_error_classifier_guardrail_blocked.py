"""OpenRouter Guardrails 403 is its own failure, not a WAF/UA problem or a key rejection (#155460).

The guardrail body ("Request blocked: prompt injection patterns detected") collided with the
generic "request blocked" WAF pattern (User-Agent advice) and, status-less on the streaming
relay, fell through to unknown/auth copy.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


_GUARDRAIL_BODY = (
    "Error code: 403 - {'error': {'message': 'Request blocked: prompt injection patterns detected', "
    "'code': 403, 'metadata': {'patterns': ['role_tag_injection'], 'provider_name': None}}}"
)


@pytest.mark.parametrize(
    "body,status",
    [
        (_GUARDRAIL_BODY, 403),
        ("403: Request blocked: prompt injection patterns detected", 403),
        ("Request blocked: prompt injection patterns detected", None),  # stream relay lost status
    ],
)
def test_guardrail_block_is_its_own_nonretryable_verdict(body, status):
    result = classify_api_error(_APIError(body, status), provider="openrouter")
    assert result.reason == FailoverReason.guardrail_blocked
    assert result.retryable is False and result.should_fallback is True
    assert result.should_rotate_credential is False and result.is_auth is False


def test_waf_and_auth_shapes_keep_their_own_reasons():
    waf = classify_api_error(_APIError("Error code: 403 - Your request was blocked.", 403), provider="openrouter")
    assert waf.reason == FailoverReason.upstream_blocked
    auth = classify_api_error(_APIError("<html>Forbidden</html>", 403), provider="openrouter")
    assert auth.reason == FailoverReason.auth
