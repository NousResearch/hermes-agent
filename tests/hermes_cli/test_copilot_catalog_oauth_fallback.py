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

    def test_skips_pool_entry_that_fails_to_exchange(self):
        """If the first entry won't exchange, try the next — an unsupported pool[0]
        must not wedge a later valid entry (Copilot review #16868 finding)."""
        attempts: list[str] = []

        def fake_exchange(raw_token: str):
            attempts.append(raw_token)
            if raw_token == "gho_unsupported_account":
                raise ValueError("Copilot token exchange failed: HTTP 401")
            return ("tid_from_second", 1234567890.0)

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                {"access_token": "gho_unsupported_account"},
                {"access_token": "gho_valid_token"},
            ],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            side_effect=fake_exchange,
        ):
            assert _resolve_copilot_catalog_api_key() == "tid_from_second"
            assert attempts == ["gho_unsupported_account", "gho_valid_token"]

    def test_all_pool_entries_fail_exchange_returns_empty(self):
        """All exchanges fail → return "" so the caller falls back to curated."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": ""},
        ), patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                {"access_token": "gho_expired_a"},
                {"access_token": "gho_expired_b"},
            ],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            side_effect=ValueError("Copilot token exchange failed"),
        ):
            assert _resolve_copilot_catalog_api_key() == ""

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
