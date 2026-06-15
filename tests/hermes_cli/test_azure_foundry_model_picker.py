"""Tests for Azure Foundry integration in the /model picker (#27989).

`provider_model_ids("azure-foundry")` used to fall straight through to the
static `_PROVIDER_MODELS["azure-foundry"] = []` table, so the in-app
``/model azure-foundry`` picker reported "0 models" even when the user's
Foundry resource exposed many deployments.

This file pins the live-discovery branch added to ``hermes_cli.models``,
which routes through the runtime credential resolver
(``hermes_cli.runtime_provider._resolve_azure_foundry_runtime``) so the
picker probes the exact resource inference will hit and honours BOTH auth
modes:

  * **API key** — ``AZURE_FOUNDRY_API_KEY`` (``.env`` or process env). The
    resolved string key is forwarded to
    ``hermes_cli.azure_detect._probe_openai_models``.
  * **Microsoft Entra ID** (``model.auth_mode: entra_id``) — the resolver
    returns a *callable* token provider instead of a string. The picker
    must detect the callable and forward it via ``token_provider=`` so the
    probe mints a fresh bearer JWT (regression for the keyless path
    flagged in review of #28006).

Any failure (missing credentials, ``azure-identity`` not installed,
network error, empty list) must fall back to the static (empty) catalog
without raising.

No real Azure endpoint is contacted — every test stubs the probe and/or
the runtime resolver.
"""

from __future__ import annotations

from unittest.mock import patch


_FAKE_FOUNDRY_DEPLOYMENTS = [
    "gpt-5.4",
    "gpt-5.3-codex",
    "kimi-k2.6",
    "deepseek-v4-pro",
    "grok-4.3",
]


# ---------------------------------------------------------------------------
# API-key auth — exercises the real runtime resolver end-to-end
# ---------------------------------------------------------------------------


class TestProviderModelIdsAzureFoundryApiKey:
    """`provider_model_ids("azure-foundry")` populates from a live probe."""

    def test_returns_live_discovered_ids_when_credentials_present(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://r.openai.azure.com/openai/v1")

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, list(_FAKE_FOUNDRY_DEPLOYMENTS)),
        ) as probe:
            ids = provider_model_ids("azure-foundry")

        assert ids == _FAKE_FOUNDRY_DEPLOYMENTS
        probe.assert_called_once()
        # API-key mode forwards the resolved string key positionally and
        # leaves token_provider unset.
        called_base, called_key = probe.call_args.args
        assert called_base == "https://r.openai.azure.com/openai/v1"
        assert called_key == "az-secret"
        assert probe.call_args.kwargs.get("token_provider") is None

    def test_prefers_config_base_url_over_env_var(self, monkeypatch):
        """Picker must target the same resource inference would (config wins)."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://env.example/v1")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "model": {
                    "provider": "azure-foundry",
                    "base_url": "https://config.example/openai/v1",
                }
            },
        )

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, ["gpt-5.4"]),
        ) as probe:
            ids = provider_model_ids("azure-foundry")

        assert ids == ["gpt-5.4"]
        called_base, _ = probe.call_args.args
        assert called_base == "https://config.example/openai/v1"

    def test_reads_api_key_from_dotenv_when_env_missing(self, monkeypatch):
        """`.env` API keys must be honoured even when the process env is empty."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://r.openai.azure.com/openai/v1")
        monkeypatch.setattr(
            "hermes_cli.config.get_env_value",
            lambda key: "dotenv-secret" if key == "AZURE_FOUNDRY_API_KEY" else "",
        )

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, ["gpt-5.4"]),
        ) as probe:
            ids = provider_model_ids("azure-foundry")

        assert ids == ["gpt-5.4"]
        _, called_key = probe.call_args.args
        assert called_key == "dotenv-secret"


# ---------------------------------------------------------------------------
# Microsoft Entra ID auth — the keyless path teknium flagged in review
# ---------------------------------------------------------------------------


class TestProviderModelIdsAzureFoundryEntraId:
    """Entra ID resolves to a callable token provider, not a string key.

    The picker must route that callable to the probe via ``token_provider=``
    so keyless (``model.auth_mode: entra_id``) users still see their
    deployments instead of an empty picker.
    """

    def test_entra_id_forwards_token_provider_to_probe(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        def sentinel_token_provider() -> str:
            return "fresh-entra-jwt"

        def _fake_runtime(*, requested_provider, model_cfg, **_kw):
            assert requested_provider == "azure-foundry"
            return {
                "provider": "azure-foundry",
                "api_mode": "chat_completions",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "api_key": sentinel_token_provider,
                "auth_mode": "entra_id",
                "source": "entra_id",
            }

        monkeypatch.setattr(
            "hermes_cli.runtime_provider._resolve_azure_foundry_runtime",
            _fake_runtime,
        )

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, list(_FAKE_FOUNDRY_DEPLOYMENTS)),
        ) as probe:
            ids = provider_model_ids("azure-foundry")

        assert ids == _FAKE_FOUNDRY_DEPLOYMENTS
        probe.assert_called_once()
        # The callable must be passed as token_provider; the positional
        # api_key must NOT be the callable (the OpenAI SDK contract differs
        # from the manual-probe contract).
        called_base, called_key = probe.call_args.args
        assert called_base == "https://r.openai.azure.com/openai/v1"
        assert called_key == ""
        assert probe.call_args.kwargs.get("token_provider") is sentinel_token_provider

    def test_entra_id_probe_failure_falls_back_to_static(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        def _fake_runtime(*, requested_provider, model_cfg, **_kw):
            return {
                "provider": "azure-foundry",
                "api_mode": "chat_completions",
                "base_url": "https://r.openai.azure.com/openai/v1",
                "api_key": lambda: "fresh-entra-jwt",
                "auth_mode": "entra_id",
                "source": "entra_id",
            }

        monkeypatch.setattr(
            "hermes_cli.runtime_provider._resolve_azure_foundry_runtime",
            _fake_runtime,
        )

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            side_effect=RuntimeError("token chain exhausted"),
        ):
            ids = provider_model_ids("azure-foundry")

        assert ids == []  # graceful fallback, no AuthError leaks into the picker


# ---------------------------------------------------------------------------
# Graceful degradation — missing creds, bad probe results, foreign providers
# ---------------------------------------------------------------------------


class TestProviderModelIdsAzureFoundryFallback:
    """Every credential/probe failure mode must yield the static ``[]``."""

    def test_ignores_config_base_url_for_non_foundry_provider(self, monkeypatch):
        """A `model.base_url` set for a different provider must not leak through."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
        # User's main provider is custom; the URL belongs to that endpoint.
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": {"provider": "custom", "base_url": "https://localhost:8000/v1"}},
        )

        with patch("hermes_cli.azure_detect._probe_openai_models") as probe:
            ids = provider_model_ids("azure-foundry")

        # No base_url for azure-foundry → resolver raises → probe never runs.
        probe.assert_not_called()
        assert ids == []

    def test_falls_back_to_static_when_api_key_missing(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://r.openai.azure.com/openai/v1")
        # Pin `.env` resolver to empty so we don't read a real key from disk.
        monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _k: "")

        with patch("hermes_cli.azure_detect._probe_openai_models") as probe:
            ids = provider_model_ids("azure-foundry")

        probe.assert_not_called()
        assert ids == []

    def test_falls_back_to_static_when_base_url_missing(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {}})

        with patch("hermes_cli.azure_detect._probe_openai_models") as probe:
            ids = provider_model_ids("azure-foundry")

        probe.assert_not_called()
        assert ids == []

    def test_falls_back_to_static_when_probe_returns_not_ok(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://r.openai.azure.com/openai/v1")

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(False, []),
        ):
            ids = provider_model_ids("azure-foundry")

        assert ids == []

    def test_falls_back_to_static_when_probe_returns_empty_list(self, monkeypatch):
        """A 200 OK with an OpenAI-shaped empty list must not block the picker."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://r.openai.azure.com/openai/v1")

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, []),
        ):
            ids = provider_model_ids("azure-foundry")

        # ok=True but ids=[]; gracefully fall through rather than returning the
        # empty live result and shadowing the static [] (semantically identical
        # here, but the contract is "fall through, not crash").
        assert ids == []

    def test_does_not_raise_when_probe_raises(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-secret")
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", "https://r.openai.azure.com/openai/v1")

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            side_effect=RuntimeError("network down"),
        ):
            ids = provider_model_ids("azure-foundry")

        assert ids == []  # graceful fallback
