"""Tests for ``is_stale_oauth_token_403`` — the xAI expired-access-token 403.

xAI answers a stale OAuth2 access token with HTTP 403 and body code
``unauthenticated:bad-credentials`` instead of the 401 every other OAuth
provider returns (issue #82052).  The predicate lets the conversation loop run
the same forced credential refresh it already runs for a 401.

The critical property is narrowness: xAI's *other* 403,
``personal-team-blocked:spending-limit``, is a billing wall that refreshing
cannot clear and must never match.
"""

import pytest

from agent.error_classifier import is_stale_oauth_token_403


def _error(status_code=None, body=None, message="api error"):
    """Build an exception shaped like the OpenAI SDK's APIStatusError."""
    exc = Exception(message)
    if status_code is not None:
        exc.status_code = status_code
    if body is not None:
        exc.body = body
    return exc


class TestStaleTokenMatches:
    def test_structured_bad_credentials_body(self):
        """The exact body from the issue report."""
        exc = _error(
            403,
            {
                "code": "unauthenticated:bad-credentials",
                "error": "The OAuth2 access token could not be validated.",
            },
        )
        assert is_stale_oauth_token_403(exc) is True

    def test_other_unauthenticated_subcode(self):
        """Any ``unauthenticated:*`` subcode is a token problem, not a plan problem."""
        exc = _error(403, {"code": "unauthenticated:token-expired"})
        assert is_stale_oauth_token_403(exc) is True

    def test_message_only_wke_marker(self):
        """No parseable body — match the stringified error the SDK gives us."""
        exc = _error(
            403,
            message=(
                "Error code: 403 - {'error': 'The OAuth2 access token could not "
                "be validated. [WKE=unauthenticated:bad-credentials]'}"
            ),
        )
        assert is_stale_oauth_token_403(exc) is True

    def test_nested_error_object_body(self):
        """Providers that wrap the code under ``error``."""
        exc = _error(
            403,
            {"error": {"code": "unauthenticated:bad-credentials", "message": "nope"}},
        )
        assert is_stale_oauth_token_403(exc) is True


class TestStaleTokenDoesNotMatch:
    def test_spending_limit_403_is_not_a_stale_token(self):
        """The regression that matters: a billing wall must not trigger refresh."""
        exc = _error(
            403,
            {"code": "personal-team-blocked:spending-limit"},
            message="Error code: 403 - spending limit reached",
        )
        assert is_stale_oauth_token_403(exc) is False

    def test_generic_403_is_not_a_stale_token(self):
        exc = _error(403, message="Error code: 403 - forbidden")
        assert is_stale_oauth_token_403(exc) is False

    @pytest.mark.parametrize("status", [401, 429, 500, None])
    def test_non_403_statuses_never_match(self, status):
        """Only 403 is ambiguous. 401 already has its own refresh branch."""
        exc = _error(
            status,
            {
                "code": "unauthenticated:bad-credentials",
                "error": "The OAuth2 access token could not be validated.",
            },
        )
        assert is_stale_oauth_token_403(exc) is False

    def test_unauthenticated_word_alone_does_not_match(self):
        """Bare 'unauthenticated' prose is not the structured xAI signal."""
        exc = _error(403, message="unauthenticated request rejected")
        assert is_stale_oauth_token_403(exc) is False
