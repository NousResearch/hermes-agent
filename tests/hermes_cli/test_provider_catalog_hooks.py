"""Provider-owned catalog seams, exercised through a fake non-Copilot profile: a profile's
``picker_catalog`` / ``static_models`` / ``probe_models`` / ``invalidate_catalog_cache`` /
``catalog_credential_files`` drive core's model catalog the way a built-in provider row does."""

from __future__ import annotations

import pytest

import providers
from hermes_cli import models
from hermes_cli import models_catalog_static as static
from providers.base import ProviderProfile


class _AcmeProfile(ProviderProfile):
    """Answers ``/models`` probes of its ``/inference`` route from its own catalog."""

    invalidated = 0

    def probe_models(self, *, base_url, api_key=None, timeout=5.0):
        if "/inference" not in base_url:
            return None
        return ["acme-owned"], "https://catalog.acme.test/models", "https://catalog.acme.test"

    def invalidate_catalog_cache(self):
        type(self).invalidated += 1


def _acme(**fields) -> _AcmeProfile:
    return _AcmeProfile(name="acme-cat", auth_type="oauth_external", hostnames=("acme.test",), **fields)


@pytest.fixture()
def only_acme(monkeypatch):
    """Make *profile* the only registered provider for the catalog lookups core performs."""
    def install(profile):
        monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile if name == profile.name else None)
        monkeypatch.setattr(providers, "list_providers", lambda: [profile])
        monkeypatch.setattr(providers, "profile_for_host", lambda url, metadata=False: profile if "acme.test" in url else None)
        return profile
    return install


def test_picker_catalog_hook_is_the_providers_live_catalog(only_acme):
    calls = []
    only_acme(_acme(picker_catalog=lambda force_refresh: calls.append(force_refresh) or ["acme-live"]))

    assert models.provider_model_ids("acme-cat", force_refresh=True) == ["acme-live"]
    assert calls == [True]


def test_picker_catalog_none_falls_through_to_the_profile_fallback(only_acme):
    only_acme(_acme(picker_catalog=lambda force_refresh: None, fallback_models=("acme-fallback",)))

    assert models.provider_model_ids("acme-cat") == ["acme-fallback"]


def test_static_models_act_as_the_providers_static_catalog_row(only_acme):
    only_acme(_acme(static_models=("acme-small", "acme-large")))

    assert models.static_catalog("acme-cat") == ["acme-small", "acme-large"]
    assert models.get_default_model_for_provider("acme-cat") == "acme-small"
    assert "acme-cat" in models.static_catalog_providers()
    assert "acme-cat" not in static._PROVIDER_MODELS  # resolved per lookup, never copied into the shared table


def test_probe_models_hook_answers_its_endpoint_and_declines_others(only_acme, monkeypatch):
    only_acme(_acme())
    fetched = []

    def fake_get_json(url, *, timeout, headers=None, **kw):
        fetched.append(url)
        return {"data": [{"id": "generic-model"}]}

    monkeypatch.setattr(models, "_get_json", fake_get_json)

    owned = models.probe_api_models("k", "https://api.acme.test/inference")
    assert (owned["models"], owned["probed_url"], owned["resolved_base_url"]) == (
        ["acme-owned"], "https://catalog.acme.test/models", "https://catalog.acme.test")
    assert fetched == []

    generic = models.probe_api_models("k", "https://api.acme.test/v1")
    assert generic["models"] == ["generic-model"]
    assert fetched == ["https://api.acme.test/v1/models"]


def test_refresh_drops_provider_owned_catalog_memos(only_acme):
    _AcmeProfile.invalidated = 0
    only_acme(_acme())

    models.clear_provider_models_cache()

    assert _AcmeProfile.invalidated == 1


def test_catalog_credential_files_rekey_cached_catalogs(only_acme, tmp_path):
    cred = tmp_path / "acme-credentials.json"
    only_acme(_acme(catalog_credential_files=(str(cred),)))
    before = models._credential_fingerprint("openrouter")

    cred.write_text("{}", encoding="utf-8")

    assert models._credential_fingerprint("openrouter") != before
