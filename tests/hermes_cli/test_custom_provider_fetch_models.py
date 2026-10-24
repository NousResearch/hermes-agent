"""Tests for live catalog probing on non-api_key and custom fetch_models provider profiles."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from providers import register_provider
from providers.base import ProviderProfile
from hermes_cli.models import _profile_live_catalog, probe_profile_catalog


class _MockCustomOAuthProfile(ProviderProfile):
    """A test profile that implements a custom fetch_models override and uses oauth_external."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fetch_calls = []

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        self.fetch_calls.append({"api_key": api_key, "base_url": base_url, "timeout": timeout})
        if api_key == "reject":
            raise RuntimeError("API error")
        return ["custom-model-1", "custom-model-2"]


class _MockDefaultOAuthProfile(ProviderProfile):
    """A test profile that does NOT override fetch_models and uses oauth_external."""
    pass


class TestCustomProviderFetchModels(unittest.TestCase):
    def test_custom_fetch_models_invoked_for_oauth_provider(self):
        """A profile with auth_type='oauth_external' and custom fetch_models is probed live."""
        profile = _MockCustomOAuthProfile(
            name="test-oauth-provider",
            auth_type="oauth_external",
            fallback_models=("fallback-1", "fallback-2"),
        )
        register_provider(profile)

        result = _profile_live_catalog("test-oauth-provider")
        self.assertIsNotNone(result)
        self.assertIn("custom-model-1", result)
        self.assertIn("custom-model-2", result)
        self.assertEqual(len(profile.fetch_calls), 1)

    def test_custom_fetch_models_without_base_url_succeeds(self):
        """A profile with no base_url but a custom fetch_models is probed rather than skipped."""
        profile = _MockCustomOAuthProfile(
            name="test-no-base-url-provider",
            auth_type="oauth_device_code",
            base_url="",  # explicitly empty
            fallback_models=("fallback-1",),
        )
        register_provider(profile)

        result = _profile_live_catalog("test-no-base-url-provider")
        self.assertEqual(result, ["fallback-1", "custom-model-1", "custom-model-2"])
        self.assertEqual(profile.fetch_calls[0]["base_url"], None)

    def test_custom_fetch_models_reads_pool_access_token(self):
        """Credentials stored in credential pool as access_token are passed to fetch_models via peek only."""
        profile = _MockCustomOAuthProfile(
            name="test-pool-cred-provider",
            auth_type="oauth_external",
            base_url="https://api.example.com",
            fallback_models=("fallback-1",),
        )
        register_provider(profile)

        mock_entry = SimpleNamespace(access_token="oauth-bearer-token-123", runtime_api_key="", api_key="")
        mock_pool = MagicMock()
        mock_pool.has_credentials.return_value = True
        mock_pool.peek.return_value = mock_entry

        with patch("agent.credential_pool.load_pool", return_value=mock_pool):
            result = _profile_live_catalog("test-pool-cred-provider")

        self.assertEqual(result, ["fallback-1", "custom-model-1", "custom-model-2"])
        self.assertEqual(profile.fetch_calls[0]["api_key"], "oauth-bearer-token-123")
        self.assertEqual(profile.fetch_calls[0]["base_url"], "https://api.example.com")
        mock_pool.peek.assert_called_once()
        mock_pool.select.assert_not_called()

    def test_multi_credential_pool_rotation_untouched_by_catalog_probe(self):
        """Probing models does not advance rotation, change current entry, or increment request_count."""
        profile = _MockCustomOAuthProfile(
            name="test-multi-cred-provider",
            auth_type="oauth_external",
            fallback_models=("fallback-1",),
        )
        register_provider(profile)

        from agent.credential_pool import CredentialPool, PooledCredential, AUTH_TYPE_OAUTH

        cred1 = PooledCredential(
            provider="test-multi-cred-provider",
            id="cred-1",
            label="Account 1",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual",
            access_token="token-1",
            request_count=0,
        )
        cred2 = PooledCredential(
            provider="test-multi-cred-provider",
            id="cred-2",
            label="Account 2",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual",
            access_token="token-2",
            request_count=0,
        )

        with patch("agent.credential_pool.get_pool_strategy", return_value="round_robin"):
            pool = CredentialPool(provider="test-multi-cred-provider", entries=[cred1, cred2])

        initial_peek_id = pool.peek().id
        self.assertEqual(initial_peek_id, "cred-1")

        with patch("agent.credential_pool.load_pool", return_value=pool):
            result = _profile_live_catalog("test-multi-cred-provider")

        self.assertIn("custom-model-1", result)
        # Verify request counts are completely untouched
        self.assertEqual(cred1.request_count, 0)
        self.assertEqual(cred2.request_count, 0)
        # Verify rotation order is completely untouched (peek is still cred-1)
        self.assertEqual(pool.peek().id, "cred-1")

    def test_custom_fetch_models_failure_falls_back(self):
        """When custom fetch_models raises, it degrades gracefully to fallback_models."""
        profile = _MockCustomOAuthProfile(
            name="test-failing-provider",
            auth_type="oauth_external",
            fallback_models=("fallback-1", "fallback-2"),
        )
        register_provider(profile)

        mock_entry = SimpleNamespace(access_token="reject", runtime_api_key="", api_key="")
        mock_pool = MagicMock()
        mock_pool.has_credentials.return_value = True
        mock_pool.peek.return_value = mock_entry

        with patch("agent.credential_pool.load_pool", return_value=mock_pool):
            result = _profile_live_catalog("test-failing-provider")

        self.assertEqual(list(result), ["fallback-1", "fallback-2"])

    def test_default_oauth_profile_without_override_keeps_fallback_models(self):
        """Non-api_key profiles without a fetch_models override still safely fall back."""
        profile = _MockDefaultOAuthProfile(
            name="test-default-oauth-provider",
            auth_type="oauth_external",
            fallback_models=("static-1", "static-2"),
        )
        register_provider(profile)

        result = _profile_live_catalog("test-default-oauth-provider")
        self.assertEqual(list(result), ["static-1", "static-2"])

    def test_explicit_has_custom_fetch_flag_controls_probe(self):
        """An explicit has_custom_fetch=False overrides subclass inheritance, and True enables probe."""
        # Subclass of Custom that opts out
        class _OptOutSubclass(_MockCustomOAuthProfile):
            has_custom_fetch = False

        opt_out = _OptOutSubclass(
            name="test-opt-out-provider",
            auth_type="oauth_external",
            fallback_models=("fallback-opt-out",),
        )
        register_provider(opt_out)
        self.assertEqual(list(_profile_live_catalog("test-opt-out-provider")), ["fallback-opt-out"])
        self.assertEqual(len(opt_out.fetch_calls), 0)


if __name__ == "__main__":
    unittest.main()
