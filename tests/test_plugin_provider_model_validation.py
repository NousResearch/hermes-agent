"""Tests for profile-based model validation of plugin providers (#101705).

A ``kind: model-provider`` plugin can override ``fetch_models()`` to return
its real catalog from a provider-specific endpoint, and the ``/model`` picker
surfaces that catalog via ``provider_model_ids()``. ``validate_requested_model()``
must validate against the same profile catalog instead of the generic
``GET {base}/v1/models`` probe: on hosts that front a shared gateway the probe
returns HTTP 200 with a *different* product catalog (the gateway's aggregator
dump), which used to hard-reject every model the picker itself offered.
"""

import dataclasses
from unittest.mock import patch

import pytest

from hermes_cli.models import validate_requested_model
from providers.base import ProviderProfile

# Simulates the shared-gateway dump: HTTP 200 with hundreds of foreign-vendor
# models and none of the plugin's subscription SKUs.
GATEWAY_DUMP = [
    "moonshotai/kimi-k3",
    "z-ai/glm-5.2",
    "z-ai/glm-5.3",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.6",
]

# The plugin's own catalog (what fetch_models() returns today via
# provider_model_ids(), including curated fallbacks).
PLUGIN_CATALOG = [
    "cline-pass/kimi-k3",
    "cline-pass/glm-5.3",
    "cline-pass/claude-sonnet-4.6",
]


def _fake_profile():
    # ClinePass-shaped: overrides fetch_models() to read a provider-specific
    # subscription-models endpoint, exactly like the community plugin.
    @dataclasses.dataclass
    class _ClinePassProfile(ProviderProfile):
        name: str = "clinepass"
        display_name: str = "ClinePass"
        auth_type: str = "api_key"
        base_url: str = "https://api.cline.bot/api/v1"
        fallback_models: tuple = ("cline-pass/kimi-k3",)

        def fetch_models(self, **kw):
            return list(PLUGIN_CATALOG)

    return _ClinePassProfile()


class TestPluginProviderModelValidation:
    """validate_requested_model() honors the plugin profile catalog."""

    @pytest.fixture(autouse=True)
    def _plugin_env(self):
        with patch(
            "providers.get_provider_profile", return_value=_fake_profile()
        ), patch(
            "hermes_cli.models.provider_model_ids",
            side_effect=lambda provider, **kw: list(PLUGIN_CATALOG),
        ) as _mock_catalog, patch(
            "hermes_cli.models.fetch_api_models",
            return_value=list(GATEWAY_DUMP),
        ) as _mock_fetch:
            yield _mock_catalog, _mock_fetch

    def test_plugin_catalog_model_accepted(self, _plugin_env):
        # The subscription SKU is absent from the gateway dump — before the
        # fix the generic probe hard-rejected it with foreign suggestions.
        result = validate_requested_model(
            "cline-pass/kimi-k3", "clinepass", api_key="sk-test"
        )
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_plugin_catalog_case_insensitive(self, _plugin_env):
        result = validate_requested_model(
            "CLINE-PASS/KIMI-K3", "clinepass", api_key="sk-test"
        )
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_plugin_catalog_rejects_gateway_dump_model(self, _plugin_env):
        # A usage-billing ID from the shared gateway is NOT one of the
        # plugin's subscription SKUs — it must be rejected against the
        # profile catalog, without suggesting the gateway's foreign models.
        result = validate_requested_model(
            "anthropic/claude-sonnet-4.6", "clinepass", api_key="sk-test"
        )
        assert result["accepted"] is False
        assert result["persist"] is False
        assert result["recognized"] is False
        assert "ClinePass" in result["message"]
        # Suggestions (if any) must come from the profile catalog, never the
        # gateway dump.
        if "Similar models" in result["message"]:
            assert "moonshotai/" not in result["message"]
            assert "cline-pass/" in result["message"]

    def test_plugin_catalog_typo_auto_corrects_within_catalog(self, _plugin_env):
        # "cline-pass/kimi-k" is a 0.97 ratio match for the catalog entry but
        # only ~0.4 against every gateway model, so the correction must come
        # from the profile catalog.
        result = validate_requested_model(
            "cline-pass/kimi-k", "clinepass", api_key="sk-test"
        )
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result["corrected_model"] == "cline-pass/kimi-k3"

    def test_plugin_catalog_suggestions_from_catalog(self, _plugin_env):
        # "cline-pass/glm" is a ~0.87 ratio match for cline-pass/glm-5.3:
        # above the 0.5 suggestion cutoff, below the 0.9 auto-correct cutoff —
        # suggests catalog entries only.
        result = validate_requested_model(
            "cline-pass/glm", "clinepass", api_key="sk-test"
        )
        assert result["accepted"] is False
        assert "cline-pass/glm-5.3" in result["message"]
        assert "moonshotai/" not in result["message"]


class TestPluginProfileFallbackPaths:
    """Fail-open: without a usable profile catalog the generic probe stays."""

    def _run(self, requested="cline-pass/kimi-k3"):
        return validate_requested_model(requested, "clinepass", api_key="sk-test")

    def test_no_profile_uses_generic_probe(self):
        # No registered profile — the old generic /v1/models path applies.
        with patch("providers.get_provider_profile", return_value=None), patch(
            "hermes_cli.models.fetch_api_models", return_value=list(GATEWAY_DUMP)
        ):
            result = self._run()
        assert result["accepted"] is False
        assert result["recognized"] is False

    def test_profile_error_uses_generic_probe(self):
        # Profile resolution blowing up must not brick model switching.
        with patch(
            "providers.get_provider_profile", side_effect=RuntimeError("boom")
        ), patch(
            "hermes_cli.models.fetch_api_models", return_value=list(GATEWAY_DUMP)
        ):
            result = self._run()
        assert result["accepted"] is False

    def test_empty_catalog_uses_generic_probe(self):
        # provider_model_ids() with no live fetch and no fallbacks returns
        # [] — nothing authoritative to validate against, keep the probe.
        with patch(
            "providers.get_provider_profile", return_value=_fake_profile()
        ), patch(
            "hermes_cli.models.provider_model_ids", return_value=[]
        ), patch(
            "hermes_cli.models.fetch_api_models", return_value=list(GATEWAY_DUMP)
        ):
            result = self._run()
        assert result["accepted"] is False

    def test_default_catalog_profile_uses_generic_probe(self):
        # A profile that neither overrides fetch_models() nor sets models_url
        # has no dedicated catalog endpoint — its listing comes from the same
        # source as the generic probe, so the probe path (with its
        # auto-correct and variant handling) must keep applying.
        default_profile = ProviderProfile(
            name="plain",
            auth_type="api_key",
            base_url="https://api.plain.test/v1",
        )
        with patch(
            "providers.get_provider_profile", return_value=default_profile
        ), patch(
            "hermes_cli.models.fetch_api_models", return_value=["plain/alpha"]
        ):
            result = validate_requested_model(
                "plain/alpha", "plain", api_key="sk-test"
            )
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_models_url_profile_owns_catalog(self):
        # models_url decouples the catalog endpoint from the inference base
        # URL, so the profile catalog is authoritative even without a
        # fetch_models() override.
        models_url_profile = ProviderProfile(
            name="plain",
            auth_type="api_key",
            base_url="https://api.plain.test/v1",
            models_url="https://api.plain.test/v2/catalog",
        )
        with patch(
            "providers.get_provider_profile", return_value=models_url_profile
        ), patch(
            "hermes_cli.models.provider_model_ids", return_value=["plain/alpha"]
        ), patch(
            "hermes_cli.models.fetch_api_models", return_value=list(GATEWAY_DUMP)
        ):
            result = validate_requested_model(
                "plain/alpha", "plain", api_key="sk-test"
            )
        assert result["accepted"] is True
        assert result["recognized"] is True
