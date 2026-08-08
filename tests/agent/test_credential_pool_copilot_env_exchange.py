"""Copilot env-seeding must persist the exchanged token, not the raw one.

``load_pool("copilot")`` runs two seeders in sequence against the *same*
``env:COPILOT_GITHUB_TOKEN`` source key:

1. ``_seed_from_singletons`` exchanges the raw ``ghu_``/``gho_`` GitHub token
   for a short-lived ``tid=...`` Copilot API token and records the
   account-specific endpoint advertised by the exchange (business/enterprise
   accounts get a dedicated proxy, e.g. ``api.enterprise.githubcopilot.com``).
2. ``_seed_from_env`` then upserts the same source key.

Because both write the same entry, step 2 used to clobber step 1's exchanged
credential with the *raw* GitHub token plus the generic
``api.githubcopilot.com`` base URL. GitHub rejects that pairing
intermittently with HTTP 403, so Copilot requests failed roughly a third of
the time while ``auth.json`` looked superficially healthy.

These tests pin the contract between the two seeders: whichever order they
run in, the persisted copilot entry carries the exchanged token and the
endpoint the exchange advertised.
"""

from unittest.mock import patch

import agent.credential_pool as credential_pool


RAW_TOKEN = "ghu_EXAMPLERAWTOKEN"
EXCHANGED_TOKEN = "tid=EXAMPLE;exp=123;sku=enterprise"
ENTERPRISE_URL = "https://api.enterprise.githubcopilot.com"
GENERIC_URL = "https://api.githubcopilot.com"


def _seed_env(entries):
    """Run the env seeder with the copilot token visible and no suppression."""
    with patch.object(
        credential_pool, "_get_secret", lambda key, default="": (
            RAW_TOKEN if key == "COPILOT_GITHUB_TOKEN" else default
        )
    ), patch.object(
        credential_pool, "load_env", return_value={}
    ), patch(
        "hermes_cli.auth.is_source_suppressed", return_value=False
    ), patch(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        return_value=(EXCHANGED_TOKEN, ENTERPRISE_URL),
    ):
        return credential_pool._seed_from_env("copilot", entries)


class TestCopilotEnvSeedingExchangesToken:
    def test_env_seeder_persists_exchanged_token_and_endpoint(self):
        """The env seeder must exchange, not store the raw GitHub token."""
        entries = []
        _seed_env(entries)

        assert len(entries) == 1
        assert entries[0].access_token == EXCHANGED_TOKEN
        assert entries[0].base_url == ENTERPRISE_URL

    def test_env_seeder_does_not_clobber_singleton_seed(self):
        """Running after the singleton seeder must not downgrade the entry.

        This is the actual failure mode: ``load_pool`` calls the singleton
        seeder first, then the env seeder, and the second write won.
        """
        entries = []
        with patch(
            "hermes_cli.copilot_auth.resolve_copilot_token",
            return_value=(RAW_TOKEN, "COPILOT_GITHUB_TOKEN"),
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value=(EXCHANGED_TOKEN, ENTERPRISE_URL),
        ), patch(
            "hermes_cli.auth.is_source_suppressed", return_value=False
        ):
            credential_pool._seed_from_singletons("copilot", entries)

        assert entries and entries[0].access_token == EXCHANGED_TOKEN

        _seed_env(entries)

        assert len(entries) == 1, "seeders must share one entry, not fork it"
        assert entries[0].access_token == EXCHANGED_TOKEN
        assert entries[0].base_url == ENTERPRISE_URL
        assert not entries[0].access_token.startswith("ghu_")
        assert entries[0].base_url != GENERIC_URL

    def test_explicit_base_url_env_var_still_wins(self):
        """A user-set COPILOT_API_BASE_URL keeps precedence over the exchange."""
        override = "https://copilot.internal.example.com"
        entries = []
        with patch.object(
            credential_pool, "_get_secret", lambda key, default="": {
                "COPILOT_GITHUB_TOKEN": RAW_TOKEN,
                "COPILOT_API_BASE_URL": override,
            }.get(key, default)
        ), patch.object(
            credential_pool, "load_env", return_value={}
        ), patch(
            "hermes_cli.auth.is_source_suppressed", return_value=False
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value=(EXCHANGED_TOKEN, ENTERPRISE_URL),
        ):
            credential_pool._seed_from_env("copilot", entries)

        assert entries[0].base_url == override
        assert entries[0].access_token == EXCHANGED_TOKEN

    def test_failed_exchange_falls_back_to_raw_token(self):
        """A failed exchange must still seed a usable entry, not crash.

        ``get_copilot_api_token`` returns ``(raw_token, None)`` when the
        exchange fails, which is the pre-existing contract for individual
        accounts that don't need exchange.
        """
        entries = []
        with patch.object(
            credential_pool, "_get_secret", lambda key, default="": (
                RAW_TOKEN if key == "COPILOT_GITHUB_TOKEN" else default
            )
        ), patch.object(
            credential_pool, "load_env", return_value={}
        ), patch(
            "hermes_cli.auth.is_source_suppressed", return_value=False
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            return_value=(RAW_TOKEN, None),
        ):
            credential_pool._seed_from_env("copilot", entries)

        assert entries[0].access_token == RAW_TOKEN
        assert entries[0].base_url == GENERIC_URL

    def test_other_providers_unaffected(self):
        """Non-copilot providers must not gain a token-exchange step."""
        entries = []
        with patch.object(
            credential_pool, "_get_secret", lambda key, default="": (
                "sk-openrouter-example" if key == "OPENROUTER_API_KEY" else default
            )
        ), patch.object(
            credential_pool, "load_env", return_value={}
        ), patch(
            "hermes_cli.auth.is_source_suppressed", return_value=False
        ), patch(
            "hermes_cli.copilot_auth.get_copilot_api_token"
        ) as mock_exchange:
            credential_pool._seed_from_env("openrouter", entries)

        mock_exchange.assert_not_called()
