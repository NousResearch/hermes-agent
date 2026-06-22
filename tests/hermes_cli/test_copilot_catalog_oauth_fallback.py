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
    def test_env_var_token_wins_over_pool(self, monkeypatch):
        """Env-resolved token still short-circuits the pool fallback."""
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_env_token")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch("hermes_cli.auth.read_credential_pool") as mock_pool, patch(
            "hermes_cli.copilot_auth._try_gh_cli_token"
        ) as mock_gh, patch(
            "hermes_cli.copilot_auth.exchange_copilot_token"
        ) as mock_exchange:
            assert _resolve_copilot_catalog_api_key() == "gho_env_token"

        mock_pool.assert_not_called()
        mock_gh.assert_not_called()
        mock_exchange.assert_not_called()

    def test_d08_pool_audit_records_skipped_invalid_entries_and_gh_fallback(
        self, monkeypatch
    ):
        """D-08: skipped invalid pool entries must remain visible in the audit."""
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        pool_entries = [
            "not-a-dict",
            {"label": "no-token-here"},
            {"access_token": ""},
            {"access_token": "ghp_classic_pat"},
        ]

        with patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=pool_entries,
        ), patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ):
            audit = resolve_copilot_identity_audit(
                include_credential_pool=True,
                exchange_pool_tokens=True,
            )
            resolved = _resolve_copilot_catalog_api_key()

        assert audit.token == "gho_from_cli"
        assert audit.source == "gh auth token"
        assert audit.source_kind == "gh_auth"
        assert [skip.source for skip in audit.skipped_sources] == [
            "credential_pool:copilot[0]",
            "credential_pool:copilot[1]",
            "credential_pool:copilot[2]",
            "credential_pool:copilot[3]",
        ]
        assert any(
            "Non-dict credential pool entry" in skip.reason
            for skip in audit.skipped_sources
        )
        assert any(
            "Missing access_token" in skip.reason
            for skip in audit.skipped_sources
        )
        assert any(
            "Classic Personal Access Tokens" in skip.reason
            for skip in audit.skipped_sources
        )
        assert resolved == "gho_from_cli"

    def test_d09_pool_token_wins_before_gh_auth(self, monkeypatch):
        """D-09: the shared identity helper should prefer the pool before gh auth."""
        from hermes_cli.copilot_auth import resolve_copilot_identity_audit

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[{"access_token": "gho_pool_token"}],
        ), patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            return_value=("tid_from_pool", 1234567890.0),
        ), patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            return_value="gho_from_cli",
        ) as mock_gh:
            audit = resolve_copilot_identity_audit(
                include_credential_pool=True,
                exchange_pool_tokens=True,
            )
            resolved = _resolve_copilot_catalog_api_key()

        assert audit.token == "tid_from_pool"
        assert audit.source == "credential_pool:copilot[0]"
        assert audit.source_kind == "credential_pool"
        assert resolved == "tid_from_pool"
        mock_gh.assert_not_called()
