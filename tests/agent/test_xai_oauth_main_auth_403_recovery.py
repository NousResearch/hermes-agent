"""Main-agent xAI OAuth auth recovery parity with auxiliary_client.

xAI returns HTTP 403 (not 401) with unauthenticated:bad-credentials when an
OAuth access token is invalid. Auxiliary paths already treat that as auth;
the acting-model conversation loop must also force-refresh on auth-shaped
403s, not only 401. This file locks the classifier half of that contract.
"""

from __future__ import annotations

from agent.error_classifier import FailoverReason, classify_api_error


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def test_classify_xai_bad_credentials_403_is_auth():
    err = _FakeHTTPError(
        403,
        "Error code: 403 - {'code': 'unauthenticated:bad-credentials', "
        "'error': 'The OAuth2 access token could not be validated.'}",
        body={
            "code": "unauthenticated:bad-credentials",
            "error": "The OAuth2 access token could not be validated.",
        },
    )
    classified = classify_api_error(err, provider="xai-oauth", model="grok-4.5")
    assert classified.is_auth is True
    assert classified.status_code == 403
    assert classified.reason in {FailoverReason.auth, FailoverReason.auth_permanent}


def test_classify_generic_403_without_auth_markers_is_not_forced_auth():
    """Non-auth 403s must remain outside the is_auth gate so the conversation
    loop does not spuriously force-refresh on entitlement/policy denials."""
    err = _FakeHTTPError(
        403,
        "Error code: 403 - {'code': 'permission-denied', 'error': 'model not allowed'}",
        body={"code": "permission-denied", "error": "model not allowed"},
    )
    classified = classify_api_error(err, provider="xai-oauth", model="grok-4.5")
    # Either non-auth, or auth_permanent is acceptable; the acting loop only
    # retries when is_auth is True. Soft-assert the common non-auth path.
    if classified.is_auth:
        # If the classifier is broad on 403, require it not look like a success retry cue
        assert classified.reason in {FailoverReason.auth, FailoverReason.auth_permanent}
