"""Provider-owned live windows must not be frozen by a legacy disk cache."""

import pytest

import providers
from agent import model_metadata as metadata
from providers.base import ProviderProfile


class CatalogProfile(ProviderProfile):
    def live_context_length(self, model, *, api_key="", base_url=""):
        self.calls.append((model, api_key, base_url))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def catalog_profile(request):
    profile = CatalogProfile(
        name="authoritative-test",
        base_url="https://catalog.test/v1",
        hostnames=("catalog.test",),
    )
    profile.use_live_model_metadata = True
    profile.answer = 32768
    profile.calls = []
    providers.list_providers()
    providers.register_provider(profile)
    metadata.save_context_length("catalog-test-model", profile.base_url, 131072)

    def cleanup():
        providers._REGISTRY.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(cleanup)
    return profile


def resolve(profile, **kwargs):
    return metadata.get_model_context_length(
        "catalog-test-model",
        base_url=profile.base_url,
        provider=profile.name,
        **kwargs,
    )


def test_opted_in_catalog_precedes_persistent_cache(catalog_profile):
    assert resolve(catalog_profile, api_key="fixture-key") == 32768
    assert catalog_profile.calls == [
        ("catalog-test-model", "fixture-key", catalog_profile.base_url),
    ]


def test_default_profile_preserves_cache_precedence(catalog_profile):
    catalog_profile.use_live_model_metadata = False
    assert resolve(catalog_profile) == 131072
    assert catalog_profile.calls == []


def test_explicit_context_override_skips_live_catalog(catalog_profile):
    assert resolve(catalog_profile, config_context_length=8192) == 8192
    assert catalog_profile.calls == []


def test_qualified_profile_bound_still_precedes_live_catalog(
    catalog_profile, monkeypatch
):
    monkeypatch.setattr(
        catalog_profile, "get_model_context_length", lambda model: 16384
    )
    assert resolve(catalog_profile) == 16384
    assert catalog_profile.calls == []


@pytest.mark.parametrize(
    "answer", [None, 0, -1, True, "32768", TimeoutError("offline")]
)
def test_failed_or_invalid_live_window_keeps_last_known_cache(catalog_profile, answer):
    catalog_profile.answer = answer
    assert resolve(catalog_profile) == 131072
    assert len(catalog_profile.calls) == 1


def test_url_inferred_profile_can_opt_in(catalog_profile):
    assert (
        metadata.get_model_context_length(
            "catalog-test-model",
            base_url=catalog_profile.base_url,
            api_key="fixture-key",
        )
        == 32768
    )
    assert len(catalog_profile.calls) == 1


def test_failed_live_probe_is_not_repeated_without_a_persistent_entry(
    catalog_profile, monkeypatch
):
    from agent import models_dev

    metadata._invalidate_cached_context_length(
        "catalog-test-model", catalog_profile.base_url
    )
    catalog_profile.answer = TimeoutError("offline")
    monkeypatch.setattr(models_dev, "lookup_models_dev_context", lambda *args: 65536)
    assert resolve(catalog_profile) == 65536
    assert len(catalog_profile.calls) == 1


def test_live_hook_receives_each_requests_endpoint_and_credential(catalog_profile):
    for url, key, window in (
        ("https://catalog.test/one/v1", "fixture-one", 32768),
        ("https://catalog.test/two/v1", "fixture-two", 65536),
    ):
        catalog_profile.answer = window
        metadata.save_context_length("catalog-test-model", url, 131072)
        assert (
            metadata.get_model_context_length(
                "catalog-test-model",
                base_url=url,
                api_key=key,
                provider=catalog_profile.name,
            )
            == window
        )
    assert catalog_profile.calls == [
        ("catalog-test-model", "fixture-one", "https://catalog.test/one/v1"),
        ("catalog-test-model", "fixture-two", "https://catalog.test/two/v1"),
    ]
