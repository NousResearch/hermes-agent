"""A provider registered after ``hermes_cli.models`` was imported still reaches the picker catalog.

``CANONICAL_PROVIDERS`` admitted plugin providers once, at import. A plugin whose imports pull
``hermes_cli.models`` in mid-discovery, or a profile registered at runtime, never reached
``list_available_providers`` / ``_PROVIDER_LABELS`` until restart: the picker twin of the auth
registry window (#102123). ``providers._sync_auth_registry`` now re-admits into both snapshots.
"""

from providers import register_provider
from providers.base import ProviderProfile


def _profile(name: str) -> ProviderProfile:
    return ProviderProfile(name=name, display_name=name, description="late plugin (direct API)")


def test_late_registered_provider_reaches_picker_catalog(monkeypatch):
    import hermes_cli.models_catalog_static as catalog
    from hermes_cli.models import list_available_providers

    # this module is imported (the snapshot exists) before the registration below
    monkeypatch.setattr(catalog, "CANONICAL_PROVIDERS", list(catalog.CANONICAL_PROVIDERS))
    monkeypatch.setattr(catalog, "_canonical_slugs", set(catalog._canonical_slugs))
    monkeypatch.setattr(catalog, "_PROVIDER_LABELS", dict(catalog._PROVIDER_LABELS))
    monkeypatch.setattr("hermes_cli.models.CANONICAL_PROVIDERS", catalog.CANONICAL_PROVIDERS)
    monkeypatch.setattr("hermes_cli.models._PROVIDER_LABELS", catalog._PROVIDER_LABELS)
    slug = "zz-late-plugin-provider"
    assert slug not in {r["id"] for r in list_available_providers()}

    import providers as registry
    registry.list_providers()  # discovery done: a registration now is a post-discovery one
    monkeypatch.setitem(registry._REGISTRY, slug, _profile(slug))  # keeps the registry scoped
    register_provider(_profile(slug))

    assert slug in {r["id"] for r in list_available_providers()}
    assert catalog._PROVIDER_LABELS[slug] == slug
    # idempotent: a second sync adds nothing
    assert catalog.sync_plugin_provider_catalog() == 0
    assert sum(1 for p in catalog.CANONICAL_PROVIDERS if p.slug == slug) == 1
