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



