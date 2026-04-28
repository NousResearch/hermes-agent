"""Catalog-API-key fallback for the Copilot ``/model`` picker.

Regression for #16708: when the user's only Copilot credential is the
OAuth ``access_token`` saved in ``auth.json`` (the device-code flow that
``hermes auth add copilot`` itself produces), the picker was silently
dropping back to a stale hardcoded list because
``_resolve_copilot_catalog_api_key`` only consulted env vars / ``gh
auth token`` and never read the credential pool.
"""

from unittest.mock import patch

from hermes_cli.models import _resolve_copilot_catalog_api_key


class TestCopilotCatalogApiKeyResolution:
    def test_env_var_token_wins_over_pool(self):
        """Env-resolved token still short-circuits the pool fallback."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "env-token"},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
        ) as mock_pool:
            assert _resolve_copilot_catalog_api_key() == "env-token"
            mock_pool.assert_not_called()

    def test_falls_back_to_pool_oauth_token(self):
        """Empty env → walk credential_pool.copilot[] for OAuth access_token."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_abc123"}],
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value="exchanged-tid_xyz",
        ):
            assert _resolve_copilot_catalog_api_key() == "exchanged-tid_xyz"

    def test_falls_back_when_env_resolution_raises(self):
        """Env path raising an exception still falls through to the pool."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            side_effect=RuntimeError("auth.json corrupt"),
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_xyz"}],
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value="exchanged-tid_xyz",
        ):
            assert _resolve_copilot_catalog_api_key() == "exchanged-tid_xyz"

    def test_skips_classic_pat_in_pool(self):
        """Classic PATs (``ghp_…``) are unsupported by the Copilot API — skip them."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "ghp_classic_pat"}],
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
        ) as mock_exchange:
            assert _resolve_copilot_catalog_api_key() == ""
            mock_exchange.assert_not_called()

    def test_skips_invalid_pool_entries(self):
        """Non-dict entries and entries without an ``access_token`` are skipped."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                "not-a-dict",
                {"label": "no-token-here"},
                {"access_token": ""},
                {"access_token": "gho_first_real_token"},
                {"access_token": "gho_should_not_reach"},
            ],
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value="exchanged-from-first",
        ) as mock_exchange:
            assert _resolve_copilot_catalog_api_key() == "exchanged-from-first"
            mock_exchange.assert_called_once_with("gho_first_real_token")

    def test_returns_empty_string_when_no_credentials_anywhere(self):
        """No env, no pool → empty string (caller falls back to curated list)."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[],
        ):
            assert _resolve_copilot_catalog_api_key() == ""

    def test_pool_failure_returns_empty_string(self):
        """If the pool read itself raises, swallow and return ""."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            side_effect=RuntimeError("auth.json locked"),
        ):
            assert _resolve_copilot_catalog_api_key() == ""
