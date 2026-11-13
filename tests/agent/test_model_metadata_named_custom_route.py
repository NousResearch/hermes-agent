"""``custom:<name>`` route ids resolve through the endpoint's profile, exactly like ``custom``.

Regression: callers that pass the raw ``model.provider`` (the dashboard's
``/api/model/info``, the gateway turn path for a named route) handed
``get_model_context_length`` the picker id ``custom:<relay>``. That label was not in the
URL-inference set, so no provider profile was consulted, the profile's
``live_context_length`` seam never fired, and a relay publishing a 131072-token window was
reported as a hardcoded family default (1M for the id that surfaced this). The same label
also blinded the models.dev capability lookup, which resolved no catalog for it.
"""

from __future__ import annotations

import pytest

import providers
from providers.base import ProviderProfile

RELAY_HOST = "named-relay.l4-test.test"
RELAY_NAME = "l4-named-relay"
RELAY_MODEL = "some-vendor-model"
RELAY_WINDOW = 131_072
BASE_URL = f"http://{RELAY_HOST}:4000/v1"
API_KEY = "relay-key"


class _LiveContextProfile(ProviderProfile):
    """A relay profile whose only authority is its own live catalog."""

    def live_context_length(self, model, *, api_key="", base_url=""):
        self.calls.append((model, api_key, base_url))
        return RELAY_WINDOW


@pytest.fixture
def relay_profile(request):
    profile = _LiveContextProfile(
        name=RELAY_NAME, base_url=BASE_URL, hostnames=(RELAY_HOST,),
    )
    profile.calls = []
    providers.list_providers()  # discovery first, so the fixture registration is not shadowed
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(RELAY_NAME, None)
        providers._PROVIDER_LIST_CACHE = None
        providers._HOST_PROFILE_MEMO.clear()

    request.addfinalizer(_cleanup)
    return profile


def _resolve(provider: str) -> int:
    from agent.model_metadata import get_model_context_length

    return get_model_context_length(
        RELAY_MODEL, base_url=BASE_URL, api_key=API_KEY, provider=provider, config_context_length=None,
    )


def test_named_custom_route_resolves_through_the_hosts_profile(relay_profile):
    """``custom:<name>`` must reach the profile the endpoint's host claims."""
    assert _resolve(f"custom:{RELAY_NAME}") == RELAY_WINDOW
    # The live seam was asked with the route's own credential and URL, once.
    assert relay_profile.calls == [(RELAY_MODEL, API_KEY, BASE_URL)]


def test_named_custom_route_matches_plain_custom_and_the_profile_name(relay_profile):
    """The named id is a route label, so all three spellings of one endpoint agree."""
    assert _resolve(f"custom:{RELAY_NAME}") == _resolve("custom") == _resolve(RELAY_NAME) == RELAY_WINDOW


@pytest.mark.parametrize(
    "provider",
    ["", "custom", "openrouter", "openai", RELAY_NAME, "custom:unclaimed-relay"],
)
def test_every_other_label_is_returned_unchanged(provider):
    """Only ``custom:<name>`` is rewritten — an explicit provider or a custom
    ``catalog_provider`` alias keyed on the route id keeps its own name."""
    from agent.model_metadata import resolve_route_provider

    assert resolve_route_provider(provider, "http://127.0.0.1:4000/v1") == provider


def test_named_custom_route_keeps_its_own_label_when_no_host_claim_exists():
    """No claim to resolve means the route id survives (its catalog alias is keyed on it)."""
    from agent.model_metadata import resolve_route_provider

    assert resolve_route_provider("custom:unclaimed-relay", "http://127.0.0.1:4000/v1") == "custom:unclaimed-relay"
    assert resolve_route_provider("custom:unclaimed-relay", "") == "custom:unclaimed-relay"


def test_capability_lookup_resolves_the_provider_the_host_claims(relay_profile):
    """The capability surface resolves the same provider id as the resolver, so catalog
    data registered under that provider is found instead of the raw ``custom:<name>``."""
    from agent.model_metadata import resolve_route_provider
    from agent.models_dev import get_model_capabilities

    config = {
        "model_overrides": {
            RELAY_NAME: {RELAY_MODEL: {"context_window": RELAY_WINDOW, "tool_call": True}},
        },
    }
    assert get_model_capabilities(
        provider=f"custom:{RELAY_NAME}", model=RELAY_MODEL, config=config,
    ) is None
    caps = get_model_capabilities(
        provider=resolve_route_provider(f"custom:{RELAY_NAME}", BASE_URL),
        model=RELAY_MODEL, config=config,
    )
    assert caps is not None
    assert caps.supports_tools is True
    assert caps.context_window == RELAY_WINDOW
