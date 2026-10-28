"""A provider plugin's alias names the same provider whether or not models.dev is reachable.

Some bundled plugins claim an alias that models.dev also uses as a provider id (``meta`` for
``meta-ai``, ``azure`` for ``azure-foundry``, ``google`` for ``gemini``, ``google-vertex`` for
``vertex``). ``resolve_provider_full`` used to answer the bare models.dev id when the catalog
loaded and the plugin id when it did not, so ``/model ... --provider meta`` persisted a
different provider depending on the network, and disagreed with the credential resolver.
"""

import pytest

from hermes_cli.auth import resolve_provider
from hermes_cli.providers import resolve_provider_full


def _entry(key: str, api: str) -> dict:
    return {"id": key, "name": key.title(), "env": [f"{key.upper().replace('-', '_')}_API_KEY"],
            "api": api, "models": {}}


# Each alias as models.dev lists it, so the catalog leg has the colliding entry to resolve to.
_CATALOG = {
    "meta": _entry("meta", "https://api.meta.ai/v1"),
    "azure": _entry("azure", "https://example.openai.azure.com/openai"),
    "google": _entry("google", "https://generativelanguage.googleapis.com/v1beta"),
    "google-vertex": _entry("google-vertex", "https://aiplatform.googleapis.com/v1"),
}

_ALIASES = {"meta": "meta-ai", "azure": "azure-foundry", "google": "gemini", "google-vertex": "vertex"}


@pytest.mark.parametrize("registry", [_CATALOG, {}], ids=["catalog", "offline"])
@pytest.mark.parametrize("alias", sorted(_ALIASES))
def test_plugin_alias_resolves_to_the_plugin_id(monkeypatch, alias, registry):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: registry)

    pdef = resolve_provider_full(alias)

    assert pdef is not None
    assert pdef.id == _ALIASES[alias]
    assert pdef.id == resolve_provider(alias)


def test_user_provider_named_like_the_alias_still_wins(monkeypatch):
    """A configured ``providers.meta`` pointing elsewhere keeps its own identity."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: _CATALOG)
    user = {"meta": {"base_url": "https://relay.example.com/v1", "api_key": "sk-test"}}

    pdef = resolve_provider_full("meta", user_providers=user)

    assert pdef is not None
    assert pdef.base_url == "https://relay.example.com/v1"
    assert pdef.id != "meta-ai"
