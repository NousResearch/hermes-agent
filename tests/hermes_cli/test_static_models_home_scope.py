"""A plugin profile's ``static_models`` belong to the home that registered it.

One process serves many profile homes. Copying the first home's ``static_models`` into the shared
static catalog made a second home with the same provider name silently default to the first home's
model, and a home without the plugin list (and switch to) a provider it does not have.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

SLUG = "acme-static"


def _plugin(models: tuple) -> str:
    return textwrap.dedent(f"""
        from providers import register_provider
        from providers.base import ProviderProfile

        register_provider(ProviderProfile(
            name="{SLUG}", display_name="Acme Static", auth_type="api_key", env_vars=("ACME_STATIC_KEY",),
            base_url="https://acme-static.invalid/v1", static_models={models!r}))
    """)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import providers
    from hermes_cli import auth_plugin_providers as app
    from hermes_cli.auth import PROVIDER_REGISTRY

    found = {}
    for name, models in (("f", ("acme-f-only",)), ("g", ("acme-g-small", "acme-g-large")), ("a", None)):
        home = found[name] = tmp_path / name
        home.mkdir()
        if models:
            plugin = home / "plugins" / "model-providers" / SLUG
            plugin.mkdir(parents=True)
            (plugin / "__init__.py").write_text(_plugin(models), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(found["a"]))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(app, "PLUGIN_MIRRORED_PROVIDERS", set(app.PLUGIN_MIRRORED_PROVIDERS))
    rows = list(dict.items(PROVIDER_REGISTRY))
    yield found
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(rows)
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _bound(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def _view():
    from hermes_cli import models
    from hermes_cli.models_catalog_static import sync_plugin_provider_catalog

    sync_plugin_provider_catalog()  # what every registration in this home runs
    return (models.get_default_model_for_provider(SLUG), models.static_catalog(SLUG),
            SLUG in models.static_catalog_providers())


def test_each_home_sees_only_its_own_static_models(homes):
    assert _bound(homes["f"], _view) == ("acme-f-only", ["acme-f-only"], True)
    assert _bound(homes["g"], _view) == ("acme-g-small", ["acme-g-small", "acme-g-large"], True)
    assert _bound(homes["a"], _view) == ("", [], False)
    assert _bound(homes["f"], _view) == ("acme-f-only", ["acme-f-only"], True)

