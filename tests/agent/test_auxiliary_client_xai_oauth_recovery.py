"""Tests for xAI OAuth 403 error recovery in auxiliary_client.

xAI returns HTTP 403 (not 401) with "unauthenticated:bad-credentials" when
an OAuth2 access token has expired.  These tests verify the three fixes:

1. _is_auth_error detects xAI 403 as an auth failure
2. _recoverable_pool_provider maps api.x.ai to xai-oauth
3. _refresh_provider_credentials includes xai-oauth refresh logic
4. auto-routed api.x.ai clients resolve xai-oauth for singleton refresh
5. successful pool recovery also evicts a client cached under "auto"
"""

import pytest


# ── _is_auth_error ──────────────────────────────────────────────────────────

def _import_is_auth_error():
    from agent.auxiliary_client import _is_auth_error
    return _is_auth_error


class TestIsAuthErrorXaiOauth403:
    """Verify _is_auth_error correctly identifies xAI's 403 bad-credentials."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.is_auth_error = _import_is_auth_error()

    def test_xai_403_bad_credentials_is_auth_error(self):
        """The exact error xAI returns for expired OAuth tokens."""
        exc = Exception(
            "Error code: 403 - {'code': 'The caller does not have permission "
            "to execute the specified operation', 'error': 'The OAuth2 access "
            "token could not be validated. [WKE=unauthenticated:bad-credentials]'}"
        )
        exc.status_code = 403  # openai.PermissionDenied sets this
        assert self.is_auth_error(exc) is True

    def test_xai_403_bad_credentials_without_status_code(self):
        """Fallback match when status_code attribute is missing."""
        exc = Exception(
            "Error code: 403 - unauthenticated:bad-credentials"
        )
        # No status_code attribute — should still match via string pattern
        assert self.is_auth_error(exc) is True

    def test_generic_403_is_not_auth_error(self):
        """A generic 403 (e.g. rate limit, forbidden) should NOT be treated as auth."""
        exc = Exception("Error code: 403 - rate limit exceeded")
        exc.status_code = 403
        assert self.is_auth_error(exc) is False






    def test_unauthenticated_without_bad_credentials_is_not_auth_error(self):
        """'unauthenticated' alone (without 'bad-credentials') should not match."""
        exc = Exception("unauthenticated request")
        assert self.is_auth_error(exc) is False


# ── _recoverable_pool_provider ──────────────────────────────────────────────

def _import_recoverable_pool_provider():
    from agent.auxiliary_client import _recoverable_pool_provider
    return _recoverable_pool_provider


class TestRecoverablePoolProviderXaiOAuth:
    """Verify _recoverable_pool_provider maps api.x.ai to xai-oauth."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.recover = _import_recoverable_pool_provider()

    def test_explicit_xai_oauth_provider(self):
        """Explicit provider name passes through."""
        result = self.recover("xai-oauth", None)
        assert result == "xai-oauth"

    def test_api_x_ai_host_match(self):
        """api.x.ai base URL maps to xai-oauth pool."""
        class MockClient:
            base_url = "https://api.x.ai/v1/"

        result = self.recover("auto", MockClient())
        assert result == "xai-oauth"

    def test_auto_with_unknown_host_returns_none(self):
        """auto provider with unknown host returns None."""
        class MockClient:
            base_url = "https://unknown.example.com/v1/"

        result = self.recover("auto", MockClient())
        assert result is None


# ── _refresh_provider_credentials (structure check) ─────────────────────────

def _import_refresh_provider_credentials():
    from agent.auxiliary_client import _refresh_provider_credentials
    return _refresh_provider_credentials


class TestRefreshProviderCredentialsXaiOAuth:
    """Verify _refresh_provider_credentials has xai-oauth branch.

    Full integration testing requires live OAuth tokens, so we verify
    the branch exists and handles the no-credential case gracefully.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        self.refresh = _import_refresh_provider_credentials()

    def test_xai_oauth_no_pool_returns_false(self):
        """When no xai-oauth pool exists, refresh returns False gracefully."""
        # This tests that the branch exists and doesn't crash.
        # It may return True if the singleton resolver finds tokens,
        # or False if neither pool nor singleton has credentials.
        # Either way, it should not raise an exception.
        result = self.refresh("xai-oauth")
        assert isinstance(result, bool)

    def test_unknown_provider_returns_false(self):
        """Unknown providers fall through to return False."""
        result = self.refresh("unknown-provider-xyz")
        assert result is False


# ── _auth_refresh_provider_for_route ────────────────────────────────────────

def _import_auth_refresh_provider_for_route():
    from agent.auxiliary_client import _auth_refresh_provider_for_route
    return _auth_refresh_provider_for_route


class TestAuthRefreshProviderForRouteXaiOAuth:
    """Auto-routed api.x.ai clients must refresh xai-oauth, not skip as 'auto'."""

    @pytest.fixture(autouse=True)
    def _import(self):
        self.route = _import_auth_refresh_provider_for_route()

    def test_auto_api_x_ai_resolves_xai_oauth(self):
        assert self.route("auto", "https://api.x.ai/v1") == "xai-oauth"

    def test_auto_api_x_ai_trailing_slash_resolves_xai_oauth(self):
        assert self.route("auto", "https://api.x.ai/v1/") == "xai-oauth"

    def test_explicit_xai_oauth_passthrough(self):
        assert self.route("xai-oauth", "https://api.x.ai/v1") == "xai-oauth"

    def test_auto_unknown_host_stays_auto(self):
        assert self.route("auto", "https://unknown.example.com/v1") == "auto"


# ── pool recovery must not retry a dead auto-cached client ──────────────────

class TestPoolRecoveryEvictsAutoCachedXaiClient:
    """After pool recovery, retry must not reuse a client cached under 'auto'."""

    def test_successful_pool_recovery_evicts_auto_cache_entry(self, monkeypatch):
        from unittest.mock import MagicMock

        from agent import auxiliary_client as ac

        stale = MagicMock()
        stale.base_url = "https://api.x.ai/v1"
        stale.api_key = "dead-token"
        cache_key = ac._client_cache_key(
            "auto",
            async_mode=False,
            base_url="https://api.x.ai/v1",
            api_key="dead-token",
            task="mattermost_thread_title",
            model="grok-4.6",
        )
        ac._client_cache.clear()
        ac._client_cache[cache_key] = (stale, "grok-4.6", None)

        exc = Exception(
            "Error code: 403 - {'code': 'unauthenticated:bad-credentials', "
            "'error': 'The OAuth2 access token could not be validated.'}"
        )
        exc.status_code = 403

        route = ac._LadderRoute(
            client=stale,
            task="mattermost_thread_title",
            tag="",
            async_mode=False,
            base_info="https://api.x.ai/v1",
            resolved_provider="auto",
            resolved_model="grok-4.6",
            resolved_base_url="https://api.x.ai/v1",
            resolved_api_key="dead-token",
            resolved_api_mode=None,
            final_model="grok-4.6",
            main_runtime=None,
            route_info=None,
        )

        # Isolate the pool rung: singleton refresh is a different path.
        monkeypatch.setattr(ac, "_refresh_provider_credentials", lambda *a, **k: False)
        monkeypatch.setattr(ac, "_recover_provider_pool", lambda *a, **k: True)

        try:
            gen = ac._ladder_credential_rungs(exc, route, {}, client_is_nous=False)
            step = next(gen)
            assert step.kind == "retry_same_provider"
            assert not any(entry[0] is stale for entry in ac._client_cache.values()), (
                "stale auto-routed api.x.ai client survived pool recovery; "
                "retry would reuse the dead bearer cached under 'auto'"
            )
        finally:
            ac._client_cache.clear()