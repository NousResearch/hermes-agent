"""A ``model.base_url`` that is the bare root of a host the configured provider claims is that
provider's own endpoint, not a relay, so live model discovery runs for it."""

import pytest

import providers
from hermes_cli import models
from hermes_cli.config import atomic_config_write
from hermes_constants import get_hermes_home
from providers.base import ProviderProfile

CLAIMING = "claim-route-test"
NEIGHBOUR = "claim-route-neighbour"


class _ClaimingProfile(ProviderProfile):
    def __init__(self, calls, **kwargs):
        super().__init__(**kwargs)
        self.calls = calls

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        self.calls.append(("relay", base_url))
        return ["relay-model"]


@pytest.fixture
def catalog_calls(request, monkeypatch):
    calls = []

    def native_catalog(force_refresh):
        calls.append(("native", None))
        return ["native-model"]

    providers.list_providers()  # discovery first, so the registrations below are not shadowed
    providers.register_provider(_ClaimingProfile(
        calls, name=CLAIMING, base_url="https://api.claimed-host.test/v1",
        hostnames=("claimed-host.test",), metadata_hostnames=("metadata-host.test",),
        picker_catalog=native_catalog,
    ))
    providers.register_provider(ProviderProfile(
        name=NEIGHBOUR, base_url="https://api.other-claim.test/v1", hostnames=("other-claim.test",),
    ))

    def _cleanup():
        providers._REGISTRY.pop(CLAIMING, None)
        providers._REGISTRY.pop(NEIGHBOUR, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    monkeypatch.setattr(models, "_api_key_credentials", lambda name: (None, None))
    return calls


def _discover(base_url):
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "model": {"provider": CLAIMING, "base_url": base_url},
    })
    return models.provider_model_ids(CLAIMING, force_refresh=True)


@pytest.mark.parametrize("base_url", [
    "https://claimed-host.test",
    "https://claimed-host.test/",
    "https://api.enterprise.claimed-host.test",
    "https://api.enterprise.claimed-host.test/",
    "HTTPS://API.CLAIMED-HOST.TEST:443/",
])
def test_bare_https_root_of_a_claimed_host_is_the_providers_own_endpoint(base_url, catalog_calls):
    assert _discover(base_url) == ["native-model"]
    assert catalog_calls == [("native", None)]


@pytest.mark.parametrize("base_url", [
    "https://claimed-host.test/relay",
    "https://claimed-host.test/v1",
    "https://claimed-host.test//",
    "https://claimed-host.test:8443",
    "http://claimed-host.test",
    "https://user@claimed-host.test",
    "https://claimed-host.test?relay=1",
    "https://claimed-host.test?",
    "https://[broken",
    "https://claimed-host.test\uff1a443",
])
def test_a_claimed_host_with_more_than_a_bare_https_root_stays_a_relay(base_url, catalog_calls):
    assert _discover(base_url) == ["relay-model"]
    assert catalog_calls == [("relay", base_url.rstrip("/"))]  # discovery probes the URL minus trailing slashes


@pytest.mark.parametrize("base_url", [
    "https://other-claim.test",
    "https://api.other-claim.test",
    "https://unclaimed.test",
    "https://claimed-host.test.example.com",
    "https://notclaimed-host.test",
    "https://metadata-host.test",
])
def test_a_host_the_provider_does_not_claim_stays_a_relay(base_url, catalog_calls):
    assert _discover(base_url) == ["relay-model"]
    assert catalog_calls == [("relay", base_url.rstrip("/"))]


def test_the_profiles_own_endpoint_is_still_not_a_relay(catalog_calls):
    assert _discover("https://api.claimed-host.test/v1") == ["native-model"]
    assert catalog_calls == [("native", None)]
