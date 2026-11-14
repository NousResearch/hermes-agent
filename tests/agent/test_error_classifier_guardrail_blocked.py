"""OpenRouter Guardrails 403 is its own failure, not a WAF/UA problem or a key rejection (#155460).

The guardrail body ("Request blocked: prompt injection patterns detected") collides with the
generic "request blocked" WAF pattern — surfacing User-Agent/extra_headers advice on the
non-streaming path — and fell to the auth copy on the streaming path. Neither is right: an
account-level content filter judged the payload, so the copy must point at the provider's
guardrail settings and the verdict must never read as auth.
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
    "body",
    [
        _GUARDRAIL_BODY,
        "403: Request blocked: prompt injection patterns detected",  # stream-relayed, no envelope
    ],
)
def test_guardrail_403_is_guardrail_blocked_not_auth(body):
    result = classify_api_error(_APIError(body, 403), provider="openrouter")
    assert result.reason == FailoverReason.guardrail_blocked
    assert result.retryable is False and result.should_fallback is True
    assert result.should_rotate_credential is False and result.is_auth is False


def test_statusless_stream_relay_is_still_guardrail_blocked():
    # The gateway/streaming relay can lose the status code; the phrase alone must decide.
    result = classify_api_error(
        _APIError("Request blocked: prompt injection patterns detected", None),
        provider="openrouter",
    )
    assert result.reason == FailoverReason.guardrail_blocked
    assert result.is_auth is False


def test_policy_refusal_semantics():
    # No stripped field or stream flag changes a guardrail's answer (resend_cannot_help).
    from agent.error_classifier import POLICY_REFUSAL_REASONS

    assert FailoverReason.guardrail_blocked in POLICY_REFUSAL_REASONS


def test_waf_and_auth_shapes_keep_their_own_reasons():
    # The generic relay WAF body still classifies as upstream_blocked…
    waf = classify_api_error(
        _APIError("Error code: 403 - Your request was blocked.", 403),
        provider="openrouter",
    )
    assert waf.reason == FailoverReason.upstream_blocked
    # …and a bare forbidden stays auth.
    auth = classify_api_error(
        _APIError("<html>Forbidden</html>", 403), provider="openrouter"
    )
    assert auth.reason == FailoverReason.auth


def test_nonretryable_copy_names_the_guardrail_fix():
    from agent.turn_failure_copy import _NONRETRYABLE_COPY

    copy = _NONRETRYABLE_COPY[FailoverReason.guardrail_blocked.value]
    lowered = copy.lower()
    assert "prompt-injection guardrail" in lowered
    assert "the key is fine" in lowered
    assert "retrying won't help" in lowered
    # The actionable dashboard path, not the WAF/User-Agent red herring.
    assert "workspaces → guardrails" in lowered
    assert "redact" in lowered and "flag" in lowered
    assert "allowlist" in lowered
    assert "user-agent" not in lowered and "extra_headers" not in lowered


def test_matched_patterns_extracted_from_error_body():
    from agent.turn_recovery import _guardrail_matched_patterns

    assert _guardrail_matched_patterns(_GUARDRAIL_BODY) == "'role_tag_injection'"
    assert (
        _guardrail_matched_patterns(
            Exception("Request blocked: prompt injection patterns detected")
        )
        == ""
    )
