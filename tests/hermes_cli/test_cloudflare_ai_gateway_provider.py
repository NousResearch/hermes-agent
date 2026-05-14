import json
from unittest.mock import MagicMock, patch

from hermes_cli import models as models_module
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import (
    CLOUDFLARE_AI_GATEWAY_MODELS,
    fetch_cloudflare_ai_gateway_models,
    provider_model_ids,
)
from providers import get_provider_profile


def _mock_urlopen(payload):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _reset_caches():
    models_module._cloudflare_ai_gateway_catalog_cache.clear()


def test_cloudflare_ai_gateway_profile_registered():
    profile = get_provider_profile("cloudflare-ai-gateway")

    assert profile is not None
    assert profile.name == "cloudflare-ai-gateway"
    assert "CLOUDFLARE_AI_GATEWAY_TOKEN" in profile.env_vars
    assert profile.supports_health_check is False


def test_cloudflare_ai_gateway_alias_resolves_profile():
    profile = get_provider_profile("cf-aig")

    assert profile is not None
    assert profile.name == "cloudflare-ai-gateway"


def test_cloudflare_ai_gateway_model_catalog_falls_back_static():
    assert provider_model_ids("cloudflare-ai-gateway") == [
        mid for mid, _ in CLOUDFLARE_AI_GATEWAY_MODELS
    ]


def test_cloudflare_ai_gateway_fetches_live_models():
    _reset_caches()
    payload = {
        "data": [
            {"id": "openai/gpt-5.5"},
            {"id": "anthropic/claude-sonnet-4-7"},
            {"id": "openai/text-embedding-3-large"},
        ]
    }

    with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
        result = fetch_cloudflare_ai_gateway_models(
            api_key="cf-token",
            base_url="https://gateway.ai.cloudflare.com/v1/acct/gw/compat",
            force_refresh=True,
        )

    assert "openai/gpt-5.5" in result
    assert "anthropic/claude-sonnet-4-7" in result
    assert "openai/text-embedding-3-large" not in result
    assert result[0] == CLOUDFLARE_AI_GATEWAY_MODELS[0][0]


def test_cloudflare_ai_gateway_keeps_aggregator_model_slug():
    model = normalize_model_for_provider(
        "anthropic/claude-sonnet-4-6",
        "cloudflare-ai-gateway",
    )

    assert model == "anthropic/claude-sonnet-4-6"


def test_cloudflare_ai_gateway_adds_vendor_for_bare_openai_model():
    model = normalize_model_for_provider("gpt-5.4-mini", "cloudflare-ai-gateway")

    assert model == "openai/gpt-5.4-mini"
