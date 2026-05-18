"""Tests for the runtime ``/model`` picker live-discovery branch on
``azure-foundry``.

The static catalog ``_PROVIDER_MODELS["azure-foundry"]`` is intentionally
empty (deployments are per-resource). Without a live-discovery branch in
``provider_model_ids``, the runtime picker shows 0 models even when the
resource has multiple deployments visible to the configured API key.

These tests cover the dispatch wiring: credentials resolution, probe
success, probe failure, and configuration-missing paths.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli import azure_detect
from hermes_cli.models import _PROVIDER_MODELS, provider_model_ids


# ---------------------------------------------------------------------------
# azure_foundry_model_ids_or_none — thin probe wrapper
# ---------------------------------------------------------------------------

class TestAzureFoundryModelIdsOrNone:
    def test_returns_ids_on_successful_probe(self):
        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, ["gpt-5.4", "kimi-k2.6"]),
        ):
            assert azure_detect.azure_foundry_model_ids_or_none(
                "https://res.openai.azure.com/openai/v1",
                "secret",
            ) == ["gpt-5.4", "kimi-k2.6"]

    def test_returns_empty_list_when_probe_returns_empty_openai_shape(self):
        # 200 + OpenAI-shaped empty list — endpoint is reachable but has no
        # deployments. Caller can still treat this as authoritative.
        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(True, []),
        ):
            assert azure_detect.azure_foundry_model_ids_or_none(
                "https://res.openai.azure.com/openai/v1",
                "secret",
            ) == []

    def test_returns_none_when_probe_fails(self):
        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            return_value=(False, []),
        ):
            assert azure_detect.azure_foundry_model_ids_or_none(
                "https://res.openai.azure.com/openai/v1",
                "secret",
            ) is None

    def test_returns_none_when_credentials_missing(self):
        assert azure_detect.azure_foundry_model_ids_or_none("", "secret") is None
        assert azure_detect.azure_foundry_model_ids_or_none(
            "https://res.openai.azure.com/openai/v1", ""
        ) is None

    def test_returns_none_when_probe_raises(self):
        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            side_effect=RuntimeError("boom"),
        ):
            assert azure_detect.azure_foundry_model_ids_or_none(
                "https://res.openai.azure.com/openai/v1",
                "secret",
            ) is None

    def test_strips_whitespace_and_trailing_slash(self):
        captured: dict[str, str] = {}

        def _fake_probe(base_url, api_key):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            return True, ["m"]

        with patch(
            "hermes_cli.azure_detect._probe_openai_models",
            side_effect=_fake_probe,
        ):
            azure_detect.azure_foundry_model_ids_or_none(
                "  https://res.openai.azure.com/openai/v1/  ",
                "  secret  ",
            )

        assert captured["base_url"] == "https://res.openai.azure.com/openai/v1"
        assert captured["api_key"] == "secret"


# ---------------------------------------------------------------------------
# provider_model_ids dispatch
# ---------------------------------------------------------------------------

class TestProviderModelIdsAzureFoundry:
    def test_dispatches_to_live_probe_when_credentials_resolve(self):
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "azure-foundry",
                "api_key": "secret",
                "base_url": "https://res.openai.azure.com/openai/v1",
                "source": "AZURE_FOUNDRY_API_KEY",
            },
        ), patch(
            "hermes_cli.azure_detect.azure_foundry_model_ids_or_none",
            return_value=["gpt-5.4", "deepseek-v4-pro"],
        ):
            assert provider_model_ids("azure-foundry") == [
                "gpt-5.4",
                "deepseek-v4-pro",
            ]

    def test_falls_back_to_static_when_probe_fails(self):
        # Before this fix the picker already returned [] for azure-foundry
        # via the curated static list — verify we preserve that behaviour
        # when discovery cannot reach the endpoint.
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "azure-foundry",
                "api_key": "secret",
                "base_url": "https://res.openai.azure.com/openai/v1",
                "source": "AZURE_FOUNDRY_API_KEY",
            },
        ), patch(
            "hermes_cli.azure_detect.azure_foundry_model_ids_or_none",
            return_value=None,
        ):
            assert provider_model_ids("azure-foundry") == list(
                _PROVIDER_MODELS["azure-foundry"]
            )

    def test_falls_back_to_static_when_credentials_missing(self):
        # No env vars, no ~/.hermes/.env entry: resolver returns empty
        # api_key and base_url. Probe must not be invoked, picker returns [].
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "azure-foundry",
                "api_key": "",
                "base_url": "",
                "source": "default",
            },
        ), patch(
            "hermes_cli.azure_detect.azure_foundry_model_ids_or_none",
        ) as probe_mock:
            assert provider_model_ids("azure-foundry") == []
            probe_mock.assert_not_called()

    def test_falls_back_to_static_when_probe_returns_empty_list(self):
        # 200 OK + empty data on /models — endpoint accepted us but
        # listed no deployments. Treat the same as failure for picker UX.
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "azure-foundry",
                "api_key": "secret",
                "base_url": "https://res.openai.azure.com/openai/v1",
                "source": "AZURE_FOUNDRY_API_KEY",
            },
        ), patch(
            "hermes_cli.azure_detect.azure_foundry_model_ids_or_none",
            return_value=[],
        ):
            assert provider_model_ids("azure-foundry") == list(
                _PROVIDER_MODELS["azure-foundry"]
            )

    def test_swallows_resolver_exception(self):
        # If credentials resolution blows up (e.g. corrupted auth state),
        # the picker must not propagate the exception — fall through to the
        # curated static list, same as before this fix.
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            side_effect=RuntimeError("auth blew up"),
        ):
            assert provider_model_ids("azure-foundry") == list(
                _PROVIDER_MODELS["azure-foundry"]
            )
