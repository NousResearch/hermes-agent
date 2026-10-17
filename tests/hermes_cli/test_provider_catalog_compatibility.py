"""Provider profiles must not redefine existing catalog/overlay routing contracts."""

from types import SimpleNamespace

import pytest

from hermes_cli import providers as catalog
from providers import get_provider_profile


@pytest.mark.parametrize("provider", [
    "meta", "openrouter", "xai", "minimax", "anthropic", "nvidia", "zai",
])
def test_catalog_and_overlay_contract_survives_registered_profile(provider, monkeypatch):
    """Use real bundled profiles: aliases and wire policy differ from the catalog layer."""
    assert get_provider_profile(provider) is not None
    info = SimpleNamespace(name="Catalog provider", env=("CATALOG_API_KEY",),
                           api="https://catalog.example.test/v1", doc="https://catalog.example.test/docs")
    monkeypatch.setattr(catalog, "_models_dev_info", lambda *_args, **_kwargs: info)
    overlay = catalog.HERMES_OVERLAYS.get(provider, catalog.HermesOverlay())

    resolved = catalog.get_provider(provider, allow_network=False)

    assert resolved is not None
    assert resolved.id == provider
    assert resolved.source == "models.dev"
    assert resolved.name == info.name
    assert resolved.transport == overlay.transport
    assert resolved.base_url == (overlay.base_url_override or info.api)
    assert resolved.base_url_env_var == overlay.base_url_env_var
    assert resolved.api_key_env_vars == tuple(dict.fromkeys((*info.env, *overlay.extra_env_vars)))
    assert resolved.is_aggregator == overlay.is_aggregator
    assert resolved.auth_type == overlay.auth_type
    assert resolved.doc == info.doc


def test_overlay_only_contract_survives_registered_profile(monkeypatch):
    """An absent catalog must not make the profile replace Hermes' own overlay."""
    assert get_provider_profile("openrouter") is not None
    monkeypatch.setattr(catalog, "_models_dev_info", lambda *_args, **_kwargs: None)

    resolved = catalog.get_provider("openrouter", allow_network=False)

    assert resolved is not None
    assert resolved.source == "hermes"
    assert resolved.id == "openrouter"
    assert resolved.transport == "openai_chat"
    assert resolved.is_aggregator is True
    assert resolved.base_url_env_var == "OPENROUTER_BASE_URL"
    assert resolved.api_key_env_vars == ()
    assert resolved.base_url == ""
