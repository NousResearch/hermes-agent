"""``detect_provider_for_model`` must not re-route a model the CURRENT provider already serves.

Regression for the OpenRouter hijack (#97487, the ``/model gpt-6-astra`` incident): a bare name
absent from the static ``_PROVIDER_MODELS`` list but present in the current provider's live catalog
fell through to the OpenRouter lookup and silently rebuilt the session on a metered aggregator.
"""

from __future__ import annotations

import pytest

from hermes_cli import models


@pytest.fixture
def live_catalog(monkeypatch):
    """Pin the disk-cached live catalog per provider and make the OpenRouter lookup loud."""
    catalogs: dict[str, list[str]] = {}
    monkeypatch.setattr(models, "cached_provider_model_ids", lambda provider, **_: list(catalogs.get(provider, [])))
    monkeypatch.setattr(models, "_find_openrouter_slug", lambda name: f"vendor/{name}")
    return catalogs


class TestCurrentProviderCatalogWins:
    @pytest.mark.parametrize("provider,model", [
        ("openai-codex", "gpt-6-astra"),       # early-access id, static Codex list lags
        ("nous", "some-portal-only-model"),    # Portal serves it, static snapshot doesn't
        ("ollama-cloud", "glm-5.3-flash"),     # static list points at zai; OpenRouter has zai/…
    ])
    def test_served_model_stays_on_current_provider(self, live_catalog, provider, model):
        live_catalog[provider] = [model]
        assert models.detect_provider_for_model(model, provider) is None

    def test_bare_name_resolves_to_current_providers_full_slug(self, live_catalog):
        live_catalog["nous"] = ["zai/glm-5.3-flash"]
        assert models.detect_provider_for_model("glm-5.3-flash", "nous") == ("nous", "zai/glm-5.3-flash")

    def test_unserved_model_still_walks_the_ladder(self, live_catalog, monkeypatch):
        from hermes_cli import models_detect

        monkeypatch.setattr(models_detect, "provider_has_credentials", lambda p: p == "openrouter")
        live_catalog["nous"] = ["hermes-4-405b"]
        assert models.detect_provider_for_model("no-such-model", "nous") == ("openrouter", "vendor/no-such-model")
