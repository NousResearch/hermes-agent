"""``restore()`` and the catalog admission guard stay consistent across generations (#123349 review)."""
from __future__ import annotations

import pytest

import hermes_cli.models  # noqa: F401  (binds _KNOWN_PROVIDER_NAMES)
import hermes_cli.models_catalog_static as mcs
from hermes_cli import provider_seam
from hermes_cli.provider_seam import GuardedDict, GuardedSet
from providers import ProviderProfile


@pytest.fixture(autouse=True)
def _seam_isolation():
    generation = provider_seam.current()
    facades = dict(provider_seam.FACADES)
    callbacks = list(provider_seam._refresh_callbacks)
    yield
    provider_seam._refresh_callbacks[:] = callbacks
    for name in set(provider_seam.FACADES) - set(facades):
        provider_seam.FACADES.pop(name)
    provider_seam.restore(generation)


def test_restore_keeps_a_container_registered_after_the_save_readable():
    saved = provider_seam.current()
    late_set = GuardedSet(__name__, "_late_probe_set", {"a"})
    late_dict = GuardedDict(__name__, "_late_probe_dict", {"a": 1, "b": 2})

    provider_seam.restore(saved)

    assert "a" in late_set and len(late_set) == 1
    assert dict(late_dict.items()) == {"a": 1, "b": 2}
    # the facade read and the C-level base storage agree
    assert sorted(dict.items(late_dict)) == sorted(late_dict.items())


def test_catalog_admission_guard_follows_restore(monkeypatch):
    import providers

    slug = "zz-seam-guard-probe"
    monkeypatch.setattr(providers, "list_providers", lambda: [ProviderProfile(name=slug)])
    saved = provider_seam.current()
    assert mcs.sync_plugin_provider_catalog() == 1

    provider_seam.restore(saved)
    assert not any(e.slug == slug for e in mcs.CANONICAL_PROVIDERS)

    # the reverted list no longer holds the slug, so the next sync admits it again
    assert mcs._plugin_provider_enters_picker(ProviderProfile(name=slug))
    assert mcs.sync_plugin_provider_catalog() == 1
    assert sum(e.slug == slug for e in mcs.CANONICAL_PROVIDERS) == 1


def test_dashboard_main_model_assignment_runs_the_refresh_hook(monkeypatch):
    import hermes_cli.config as config
    import hermes_cli.models_detect as models_detect
    from hermes_cli.web_server_config import _normalize_main_model_assignment

    slug = "zz-late-dashboard-provider"
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(models_detect, "provider_has_credentials", lambda _p: True)
    seen = []

    def publish_late(reason, name):
        seen.append((reason, name))
        if name == slug:
            provider_seam.publish({"_KNOWN_PROVIDER_NAMES": {slug}})

    provider_seam.register_refresh(publish_late)

    provider, _model = _normalize_main_model_assignment(slug, "vendor/some-model")

    assert ("typed", slug) in seen
    assert provider == slug  # not rewritten to openrouter
