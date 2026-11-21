"""A registered alias of a provider profile reads that profile's models.dev catalog.

Sessions can record their provider by one of the profile's aliases (the GitHub Copilot plugin's
sessions record ``github-copilot`` for the profile ``copilot``). ``models_dev_id_for`` matches exact
profile names only, so every catalog consumer saw those sessions' models as unknown and, for example,
``image_input_mode: auto`` sent pictures down the text path for a vision model.
"""

import pytest

import providers
from providers.base import ProviderProfile
from agent import models_dev

CATALOG = {
    "seer-1": {"id": "seer-1", "tool_call": True, "reasoning": True, "limit": {"context": 900000, "output": 64000},
               "modalities": {"input": ["text", "image"], "output": ["text"]}},
    "blind-1": {"id": "blind-1", "tool_call": True, "limit": {"context": 100000, "output": 8000},
                "modalities": {"input": ["text"], "output": ["text"]}},
}
NO_CONFIG = {"providers": {}, "custom_providers": []}


@pytest.fixture
def registry(monkeypatch):
    """Isolated provider registry, including the per-home layer (no rescans, no recorded misses)."""
    import hermes_cli.models as hm
    import hermes_cli.models_catalog_static as static
    monkeypatch.setattr(static, "CANONICAL_PROVIDERS", list(static.CANONICAL_PROVIDERS))
    monkeypatch.setattr(static, "_canonical_slugs", set(static._canonical_slugs))
    monkeypatch.setattr(static, "_PROVIDER_LABELS", dict(static._PROVIDER_LABELS))
    monkeypatch.setattr(hm, "CANONICAL_PROVIDERS", static.CANONICAL_PROVIDERS)
    monkeypatch.setattr(hm, "_PROVIDER_LABELS", static._PROVIDER_LABELS)
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_SOURCES", {})
    monkeypatch.setattr(providers, "_discovered", True)
    layer = providers._HomeLayer()
    monkeypatch.setattr(providers, "_bound_home_layer", lambda *a, **k: (layer, "fixture-home", "fixture-key"))
    monkeypatch.setattr(providers, "_refresh_home_layer", lambda *a, **k: True)
    monkeypatch.setattr(providers, "_has_named_custom_provider", lambda *a, **k: False)
    monkeypatch.setattr(models_dev, "_load_model_overrides", dict)
    served = {"fixture-mdev": CATALOG}
    seen = []

    def registry_models(mdev_id, **_):
        seen.append(mdev_id)
        return served.get(mdev_id)

    monkeypatch.setattr(models_dev, "_registry_models", registry_models)
    providers.register_provider(ProviderProfile(
        name="fixture-provider", aliases=("fixture-alias", "fixture-other", "customai"),
        models_dev_id="fixture-mdev"))
    providers.register_provider(ProviderProfile(name="bare-provider", aliases=("bare-alias",)))
    return seen


@pytest.mark.parametrize("name", ["fixture-provider", "fixture-alias", "fixture-other", "customai"])
def test_profile_name_and_aliases_read_the_same_catalog(registry, name):
    caps = models_dev.get_model_capabilities(name, "seer-1", config=NO_CONFIG)
    assert caps is not None
    assert (caps.supports_vision, caps.supports_reasoning, caps.context_window) == (True, True, 900000)
    blind = models_dev.get_model_capabilities(name, "blind-1", config=NO_CONFIG)
    assert blind is not None and blind.supports_vision is False
    assert set(registry) == {"fixture-mdev"}


def test_alias_session_routes_images_natively(registry, monkeypatch):
    from agent.image_routing import decide_image_input_mode

    monkeypatch.setattr(models_dev, "_cfg_get", lambda *keys, default=None, config=None: default)
    cfg = {"model": {"provider": "fixture-alias", "default": "seer-1"}}
    assert decide_image_input_mode("fixture-alias", "seer-1", cfg) == "native"
    assert decide_image_input_mode("fixture-alias", "blind-1", cfg) == "text"
    assert decide_image_input_mode("fixture-alias", "not-in-catalog", cfg) == "text"


@pytest.mark.parametrize("name", ["", "no-such-provider", "FIXTURE-ALIAS", "custom:fixture-alias",
                                  "Custom:customai", "custom",
                                  "bare-alias", "bare-provider"])
def test_edge_names_stay_unknown(registry, name):
    assert models_dev._models_dev_id(name, config=NO_CONFIG) is None


def test_whitespace_around_alias_is_ignored(registry):
    assert models_dev._models_dev_id(" fixture-alias ", config=NO_CONFIG) == "fixture-mdev"


@pytest.mark.parametrize("config", [
    {"providers": {"fixture-alias": {"base_url": "http://127.0.0.1:9/v1"}}},
    {"providers": {}, "custom_providers": [{"name": "fixture-alias", "base_url": "http://127.0.0.1:9/v1"}]},
])
def test_user_defined_provider_with_an_alias_name_never_borrows_the_vendor_catalog(registry, config):
    assert models_dev._models_dev_id("fixture-alias", config=config) is None
    assert models_dev._models_dev_id("fixture-other", config=config) == "fixture-mdev"  # other names unaffected


def test_configured_catalog_provider_keeps_precedence(registry, monkeypatch):
    monkeypatch.setattr(models_dev, "_configured_catalog_provider", lambda key, config=None: "other-mdev")
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda allow_network=False: {"other-mdev": {}})
    assert models_dev._models_dev_id("fixture-alias", config=NO_CONFIG) == "other-mdev"
    monkeypatch.setattr(models_dev, "_configured_catalog_provider", lambda key, config=None: "typo-mdev")
    assert models_dev._models_dev_id("fixture-alias", config=NO_CONFIG) is None


def test_registry_failure_degrades_to_unknown(registry, monkeypatch):
    def boom(name):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(providers, "get_provider_profile", boom)
    assert models_dev._models_dev_id("fixture-alias", config=NO_CONFIG) is None
