"""Focused tests for Runware first-class provider wiring."""

from unittest.mock import patch

from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS, normalize_provider
from agent.model_metadata import get_model_context_length


class TestRunwareAliases:
    def test_models_normalize_provider(self):
        assert normalize_provider("runware-ai") == "runware"
        assert normalize_provider("runwareai") == "runware"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("runware-ai") == "runware"


class TestRunwareCanonicalEntry:
    def test_canonical_provider_entry(self):
        assert "runware" in {p.slug for p in CANONICAL_PROVIDERS}


class TestRunwareProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["runware"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("RUNWARE_API_KEY",)
        assert overlay.base_url_override == "https://api.runware.ai/v1"
        assert overlay.base_url_env_var == "RUNWARE_BASE_URL"
        assert not overlay.is_aggregator

    def test_provider_label(self):
        assert _PROVIDER_LABELS["runware"] == "Runware"


class TestRunwareContextLength:
    def test_prefers_live_endpoint_metadata_over_hardcoded_defaults(self):
        # "openai-gpt-5-4" would otherwise fuzzy-match the generic "gpt-5"
        # entry in DEFAULT_CONTEXT_LENGTHS (400000) instead of Runware's
        # real 1,050,000-token window — live /models must win.
        with patch(
            "agent.model_metadata.get_cached_context_length",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value={"openai-gpt-5-4": {"context_length": 1050000}},
        ):
            result = get_model_context_length(
                "openai-gpt-5-4",
                base_url="https://api.runware.ai/v1",
                api_key="runware-test-key",
                provider="runware",
            )

        assert result == 1050000


class TestRunwareEndpointMetadataBareList:
    def test_fetch_endpoint_model_metadata_handles_bare_array(self):
        # Runware's GET /v1/models returns a raw JSON array, not the usual
        # {"data": [...]} wrapper most OpenAI-compatible servers use.
        from agent import model_metadata as mm

        mm._endpoint_model_metadata_cache.clear()
        mm._endpoint_model_metadata_cache_time.clear()

        fake_payload = [
            {
                "id": "openai-gpt-5-4",
                "context_length": 1050000,
                "pricing": {"prompt": "0.0000025", "completion": "0.000015"},
            }
        ]

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return fake_payload

        with patch("agent.model_metadata.requests.get", return_value=_Resp()):
            metadata = mm.fetch_endpoint_model_metadata(
                "https://api.runware.ai/v1", api_key="runware-test-key"
            )

        assert metadata["openai-gpt-5-4"]["context_length"] == 1050000
        assert metadata["openai-gpt-5-4"]["pricing"]["prompt"] == "0.0000025"


class TestRunwarePricing:
    def test_get_pricing_for_provider_handles_bare_array(self, monkeypatch):
        import json
        import urllib.request

        from hermes_cli.models import _pricing_cache, get_pricing_for_provider

        _pricing_cache.clear()
        monkeypatch.setenv("RUNWARE_API_KEY", "runware-test-key")
        monkeypatch.delenv("RUNWARE_BASE_URL", raising=False)

        fake_payload = [
            {"id": "openai-gpt-5-4", "pricing": {"prompt": "0.0000025", "completion": "0.000015"}},
        ]

        class _Resp:
            def read(self):
                return json.dumps(fake_payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch.object(urllib.request, "urlopen", return_value=_Resp()) as mock_urlopen:
            pricing = get_pricing_for_provider("runware", force_refresh=True)

        assert pricing == {"openai-gpt-5-4": {"prompt": "0.0000025", "completion": "0.000015"}}
        called_url = mock_urlopen.call_args[0][0].full_url
        assert called_url == "https://api.runware.ai/v1/models"

    def test_get_pricing_for_provider_returns_empty_without_key(self, monkeypatch):
        monkeypatch.delenv("RUNWARE_API_KEY", raising=False)

        from hermes_cli.models import get_pricing_for_provider

        assert get_pricing_for_provider("runware", force_refresh=True) == {}
