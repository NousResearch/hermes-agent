"""Native Copilot plan endpoints retain live model discovery, not relay fallbacks."""

import io
import json

import pytest

from hermes_cli import models
from hermes_cli.config import atomic_config_write
from hermes_constants import get_hermes_home


@pytest.fixture
def catalog_requests(monkeypatch):
    requests = []
    payload = json.dumps({"data": [{
        "id": "account-catalog-model",
        "model_picker_enabled": True,
        "capabilities": {"type": "chat"},
        "supported_endpoints": ["/chat/completions"],
    }]}).encode()

    def open_catalog(request, **kwargs):
        requests.append(request.full_url)
        assert request.get_header("Authorization") == "Bearer fixture-catalog-token"
        return io.BytesIO(payload)

    monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "fixture-catalog-token")
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", open_catalog)
    monkeypatch.setattr(models, "_github_model_catalog_cache", None)
    return requests


@pytest.mark.parametrize("provider", ["copilot", "github-copilot"])
@pytest.mark.parametrize("base_url", [
    "https://api.individual.githubcopilot.com",
    "https://api.business.githubcopilot.com",
    "https://api.enterprise.githubcopilot.com/",
    "HTTPS://API.BUSINESS.GITHUBCOPILOT.COM:443/",
])
def test_native_plan_endpoint_lists_live_account_models(provider, base_url, catalog_requests):
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "model": {"provider": provider, "base_url": base_url},
    })

    catalog = models.provider_model_ids(provider, force_refresh=True)

    assert "account-catalog-model" in catalog
    assert catalog_requests == [models.COPILOT_MODELS_URL]


@pytest.mark.parametrize("provider,base_url", [
    ("copilot", "https://api.business.githubcopilot.com/relay"),
    ("copilot", "https://api.business.githubcopilot.com//"),
    ("github-copilot", "https://api.enterprise.githubcopilot.com////"),
    ("copilot", "https://api.enterprise.githubcopilot.com/v1"),
    ("copilot", "https://api.business.githubcopilot.com:8443"),
    ("copilot", "http://api.business.githubcopilot.com"),
    ("copilot", "https://api.business.githubcopilot.com.example.com"),
    ("copilot", "https://evilgithubcopilot.com"),
    ("copilot", "https://user@api.business.githubcopilot.com"),
    ("copilot", "https://api.business.githubcopilot.com?relay=1"),
    ("copilot", "https://api.business.githubcopilot.com?"),
    ("openai-codex", "https://api.business.githubcopilot.com"),
])
def test_other_routes_do_not_contact_native_copilot_catalog(provider, base_url, catalog_requests):
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "model": {"provider": provider, "base_url": base_url},
    })

    catalog = models.provider_model_ids(provider, force_refresh=True)

    assert "account-catalog-model" not in catalog
    assert catalog_requests == []
