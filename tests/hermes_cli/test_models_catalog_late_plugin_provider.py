"""A provider registered after ``hermes_cli.models`` was imported still reaches the picker catalog.

``CANONICAL_PROVIDERS`` admitted plugin providers once, at import. A plugin whose imports pull
``hermes_cli.models`` in mid-discovery, or a profile registered at runtime, never reached
``list_available_providers`` / ``_PROVIDER_LABELS`` until restart: the picker twin of the auth
registry window (#102123). ``providers._sync_auth_registry`` now re-admits into both snapshots.
"""

import pytest

from providers import register_provider
from providers.base import ProviderProfile


def _profile(name: str) -> ProviderProfile:
    return ProviderProfile(name=name, display_name=name, description="late plugin (direct API)")


@pytest.fixture
def _seam_generation():
    # The catalog containers are seam facades: scope the test by restoring the committed
    # generation, not by swapping in plain copies the seam never publishes to.
    from hermes_cli import provider_seam

    saved = provider_seam.current()
    yield
    provider_seam.restore(saved)


def test_late_registered_provider_reaches_picker_catalog(monkeypatch, _seam_generation):
    import hermes_cli.models_catalog_static as catalog
    from hermes_cli.models import list_available_providers

    # this module is imported (the snapshot exists) before the registration below
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
