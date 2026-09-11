"""Azure Foundry in the /model picker (#27989).

Deployments are per-resource so the static catalog is intentionally empty; ``provider_model_ids``
must probe ``GET <base>/models`` through the runtime credential resolver (API key OR Entra ID
token provider) and fall back to the static ``[]`` on any failure. No real endpoint is contacted.
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


class TestProviderModelIdsAzureFoundryFallback:
    """Every credential/probe failure mode must yield the static ``[]``."""

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
