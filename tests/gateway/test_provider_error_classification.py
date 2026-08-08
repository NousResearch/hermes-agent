"""A quota cap must not be reported to chat as an authentication failure.

Regression test for the gateway path that hard-labeled every
provider-resolution failure "Provider authentication failed". A provider usage
cap (HTTP 429) reached users as a credentials error, sending operators to
re-authenticate credentials that were valid the whole time.
"""

import pytest

from gateway.platforms.api_server import _ProviderAuthResolutionError
from gateway.run import (
    _GATEWAY_REPLY_AUTH,
    _GATEWAY_REPLY_GENERIC,
    _GATEWAY_REPLY_RATE_LIMIT,
    _GATEWAY_REPLY_TEMPORARY,
    _gateway_provider_exception_reply,
    _gateway_runtime_failure_response,
    _gateway_runtime_failure_text,
)
from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE


def _wrapped(exc: Exception) -> RuntimeError:
    """Mirror how _resolve_runtime_agent_kwargs re-raises resolution failures."""
    try:
        raise RuntimeError(str(exc)) from exc
    except RuntimeError as wrapper:
        return wrapper


def _api_server_wrapped(exc: Exception) -> _ProviderAuthResolutionError:
    """Mirror api_server's extra wrapping layer.

    _create_agent() catches the RuntimeError from _resolve_runtime_agent_kwargs()
    and re-raises it as _ProviderAuthResolutionError, so the AuthError sits two
    levels down the __cause__ chain on that path.
    """
    inner = _wrapped(exc)
    try:
        raise _ProviderAuthResolutionError(str(inner)) from inner
    except _ProviderAuthResolutionError as wrapper:
        return wrapper


def _quota_error() -> AuthError:
    return AuthError(
        "Codex provider quota exhausted (429); retry after 160602s. "
        "Credentials are still valid.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )


class TestRateLimitNotMislabeled:
    def test_quota_exhaustion_reports_as_rate_limit(self):
        reply = _gateway_provider_exception_reply(_wrapped(_quota_error()))
        assert reply == _GATEWAY_REPLY_RATE_LIMIT
        assert "authentication failed" not in reply.lower()

    def test_unwrapped_quota_error_also_classified(self):
        assert _gateway_provider_exception_reply(_quota_error()) == _GATEWAY_REPLY_RATE_LIMIT


class TestCredentialFailuresStillReportAsAuth:
    @pytest.mark.parametrize("relogin_required", [True, False])
    def test_missing_credentials_report_as_auth(self, relogin_required):
        """Many genuine credential errors leave relogin_required False, so the
        classification must not be gated on that flag."""
        exc = AuthError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY.",
            provider="anthropic",
            code="missing_api_key",
            relogin_required=relogin_required,
        )
        assert _gateway_provider_exception_reply(_wrapped(exc)) == _GATEWAY_REPLY_AUTH


class TestEntitlementAndTemporary:
    def test_entitlement_error_keeps_actionable_wording(self):
        exc = AuthError(
            "Subscription credits are exhausted.",
            provider="nous",
            code="insufficient_credits",
            relogin_required=False,
        )
        reply = _gateway_provider_exception_reply(_wrapped(exc))
        # Neither a credentials problem nor a rate limit.
        assert reply not in (_GATEWAY_REPLY_AUTH, _GATEWAY_REPLY_RATE_LIMIT)
        assert "credits" in reply.lower()

    def test_temporarily_unavailable_is_its_own_category(self):
        exc = AuthError(
            "Provider is warming up.",
            provider="nous",
            code="temporarily_unavailable",
            relogin_required=False,
        )
        assert _gateway_provider_exception_reply(_wrapped(exc)) == _GATEWAY_REPLY_TEMPORARY

    def test_provider_side_server_error_is_not_a_credentials_problem(self):
        """auth.py raises code="server_error" when Nous cannot resolve an
        inference key. Nothing is wrong with the operator's credentials, so
        telling them to check credentials is the same class of mislabel."""
        exc = AuthError(
            "Failed to resolve a Nous inference API key",
            provider="nous",
            code="server_error",
        )
        assert _gateway_provider_exception_reply(_wrapped(exc)) == _GATEWAY_REPLY_GENERIC


class TestNonAuthErrors:
    @pytest.mark.parametrize(
        "message, expected",
        [
            ("Error code: 401 - invalid api key", _GATEWAY_REPLY_AUTH),
            ("Rate limited after 3 retries", _GATEWAY_REPLY_RATE_LIMIT),
            ("connection reset by peer", _GATEWAY_REPLY_GENERIC),
        ],
    )
    def test_plain_exceptions_fall_back_to_text_classification(self, message, expected):
        assert _gateway_provider_exception_reply(RuntimeError(message)) == expected


class TestTurnResponseShape:
    """Covers the value the gateway turn actually returns, not just the classifier."""

    def test_chat_surface_gets_classified_category(self):
        result = _gateway_runtime_failure_response(_wrapped(_quota_error()), "slack")
        assert result["final_response"] == _GATEWAY_REPLY_RATE_LIMIT
        assert result["api_calls"] == 0
        assert result["messages"] == [] and result["tools"] == []

    def test_programmatic_surface_keeps_raw_detail(self):
        """CLI/API/webhook callers are promised raw diagnostics."""
        result = _gateway_runtime_failure_response(_wrapped(_quota_error()), "local")
        assert "quota exhausted" in result["final_response"]
        assert result["final_response"] != _GATEWAY_REPLY_RATE_LIMIT


class TestApiServerSurface:
    """api_server has its own _ProviderAuthResolutionError handlers and never
    calls _sanitize_gateway_final_response, so it needs its own coverage."""

    @pytest.mark.parametrize("surface", ["api_server", "webhook"])
    def test_quota_not_labeled_as_auth(self, surface):
        text = _gateway_runtime_failure_text(_api_server_wrapped(_quota_error()), surface)
        assert "authentication failed" not in text.lower()
        # Raw diagnostics preserved for these programmatic surfaces.
        assert "quota exhausted" in text

    def test_session_chat_response_shape(self):
        result = _gateway_runtime_failure_response(
            _api_server_wrapped(_quota_error()), "api_server")
        assert "authentication failed" not in result["final_response"].lower()
        assert result["api_calls"] == 0
        assert result["messages"] == [] and result["tools"] == []

    def test_double_wrapped_auth_error_still_classified_for_chat(self):
        """The AuthError sits two __cause__ levels down on the api_server path;
        a chat surface must still get the rate-limit category, not auth."""
        assert _gateway_runtime_failure_text(
            _api_server_wrapped(_quota_error()), "slack") == _GATEWAY_REPLY_RATE_LIMIT

    def test_genuine_credential_failure_still_reads_as_auth_on_chat(self):
        exc = AuthError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY.",
            provider="anthropic",
            code="missing_api_key",
            relogin_required=False,
        )
        assert _gateway_runtime_failure_text(
            _api_server_wrapped(exc), "slack") == _GATEWAY_REPLY_AUTH
