"""Catalog-API-key fallback for the Copilot ``/model`` picker.

Regression for #16708: when the user's only Copilot credential is a
``gho_*`` token (typically obtained via device-code login) stored in
``auth.json`` under ``credential_pool.copilot[]`` — placed there by
``hermes auth add copilot`` or by ``_seed_from_env`` when the env var
is set in ``~/.hermes/.env`` — the picker was silently dropping back to
a stale hardcoded list because ``_resolve_copilot_catalog_api_key``
only consulted env vars / ``gh auth token`` and never read the
credential pool.
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
        """Empty env → walk credential_pool.copilot[] for an OAuth access_token."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_abc123"}],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            return_value=("tid_exchanged_xyz", 1234567890.0),
        ):
            assert _resolve_copilot_catalog_api_key() == "tid_exchanged_xyz"

    def test_falls_back_when_env_resolution_raises(self):
        """Env path raising an exception still falls through to the pool."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            side_effect=RuntimeError("auth.json corrupt"),
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_xyz"}],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            return_value=("tid_exchanged_xyz", 1234567890.0),
        ):
            assert _resolve_copilot_catalog_api_key() == "tid_exchanged_xyz"

    def test_skips_classic_pat_in_pool(self):
        """Classic PATs (``ghp_…``) are unsupported by the Copilot API — skip them."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "ghp_classic_pat"}],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
        ) as mock_exchange:
            assert _resolve_copilot_catalog_api_key() == ""
            mock_exchange.assert_not_called()

    def test_skips_invalid_pool_entries_until_first_exchangeable(self):
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
            "hermes_cli.copilot_auth.exchange_copilot_token",
            return_value=("tid_from_first", 1234567890.0),
        ) as mock_exchange:
            assert _resolve_copilot_catalog_api_key() == "tid_from_first"
            mock_exchange.assert_called_once_with("gho_first_real_token")

    def test_exchange_failure_returns_raw_token_as_fallback(self):
        """When exchange fails, get_copilot_api_token returns the raw token
        as fallback — useful for Business/Enterprise accounts whose tokens
        work directly without exchange."""

        def fake_get_token(raw_token: str):
            if raw_token == "gho_unsupported_account":
                return "gho_unsupported_account"  # raw fallback
            return "tid_from_second"

        with (
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": ""},
            ),
            patch(
                "hermes_cli.auth.read_credential_pool",
                return_value=[
                    {"access_token": "gho_unsupported_account"},
                    {"access_token": "gho_valid_token"},
                ],
            ),
            patch(
                "hermes_cli.copilot_auth.get_copilot_api_token",
                side_effect=fake_get_token,
            ),
        ):
            assert _resolve_copilot_catalog_api_key() == "gho_unsupported_account"

    def test_exchange_failure_falls_back_to_raw_token(self):
        """When exchange fails, get_copilot_api_token returns the raw token —
        the first valid pool entry's raw token is returned (not empty)."""
        with (
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": ""},
            ),
            patch(
                "hermes_cli.auth.read_credential_pool",
                return_value=[
                    {"access_token": "gho_expired_a"},
                    {"access_token": "gho_expired_b"},
                ],
            ),
            patch(
                "hermes_cli.copilot_auth.get_copilot_api_token",
                return_value="gho_fallback_token",
            ),
        ):
            assert _resolve_copilot_catalog_api_key() == "gho_fallback_token"

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
