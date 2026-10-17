"""A credential-pool 429 bench on a native provider is rate-limiting, not a missing credential.

Regression for #138816: cron preflight reported "provider credential missing" for an Anthropic
credential the pool had benched for one model, blocking the job instead of holding it.
"""

import pytest

from hermes_cli.auth import AuthError, format_auth_error, is_rate_limited_auth_error
from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE


def _pool_bench_error() -> AuthError:
    """The exact shape ``_anthropic_token_or_raise`` raises for a benched model."""
    return AuthError(
        "Anthropic credentials are rate-limited for claude-sonnet-5-5; "
        "other Claude models remain available (see `hermes auth list`).",
        provider="anthropic",
        retryable=True,
    )


def test_pool_bench_message_is_rate_limited():
    assert is_rate_limited_auth_error(_pool_bench_error()) is True


def test_pool_bench_copy_does_not_send_operator_to_re_auth():
    rendered = format_auth_error(_pool_bench_error())
    assert "re-authenticate" not in rendered


def test_codex_code_still_detected():
    err = AuthError("usage limit reached", code=CODEX_RATE_LIMITED_CODE)
    assert is_rate_limited_auth_error(err) is True


def test_missing_credential_stays_a_credential_problem():
    assert is_rate_limited_auth_error(AuthError("No Anthropic credentials found")) is False


def test_relogin_required_is_never_rate_limiting():
    err = AuthError("Anthropic credentials are rate-limited for x", relogin_required=True)
    assert is_rate_limited_auth_error(err) is False


def test_non_auth_error_is_not_rate_limiting():
    assert is_rate_limited_auth_error(RuntimeError("Anthropic credentials are rate-limited for x")) is False


@pytest.mark.parametrize("phrase", ["credentials are rate-limited for", "credential is rate-limited for"])
def test_singular_and_plural_phrasings(phrase):
    assert is_rate_limited_auth_error(AuthError(f"Anthropic {phrase} claude-sonnet-5-5")) is True
