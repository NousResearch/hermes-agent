"""Host -> profile seams: ``hostnames`` / ``metadata_hostnames`` claims and the live context hook.

A plugin profile claims the hosts that serve its API; URL-based inference (model metadata, auxiliary
pool/refresh routing) resolves them lazily through ``providers.profile_for_host`` instead of core tables.
"""

from __future__ import annotations

import pytest

import providers
from providers.base import ProviderProfile

NAME = "l4-fake-hosted"


class _LiveContextProfile(ProviderProfile):
    def live_context_length(self, model, *, api_key="", base_url=""):
        self.calls.append((model, api_key, base_url))
        return 777_000


@pytest.fixture
def fake_profile(request):
    profile = _LiveContextProfile(
        name=NAME, base_url="https://api.fake-llm.test/v1",
        hostnames=("fake-llm.test",), metadata_hostnames=("catalog.fake-meta.test",),
    )
    profile.calls = []
    providers.list_providers()  # discovery first, so the fixture registration is not shadowed
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(NAME, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    return profile


def test_profile_for_host_matches_claimed_hosts_exact_or_subdomain(fake_profile):
    assert providers.profile_for_host("https://api.fake-llm.test/v1") is fake_profile
    assert providers.profile_for_host("https://api.enterprise.fake-llm.test") is fake_profile
    assert providers.profile_for_host("fake-llm.test") is fake_profile
    # never a substring: a look-alike host must not inherit the profile
    assert providers.profile_for_host("https://fake-llm.test.attacker.invalid") is None
    assert providers.profile_for_host("https://notfake-llm.test") is None
    # metadata-only hosts need the explicit opt-in
    assert providers.profile_for_host("https://catalog.fake-meta.test/inference") is None
    assert providers.profile_for_host("https://catalog.fake-meta.test/inference", metadata=True) is fake_profile


def test_claimed_hosts_drive_url_inference_and_aux_routing(fake_profile):
    from agent.auxiliary_client import _AUTH_REFRESH_PROVIDER_BY_HOST, _POOL_PROVIDER_BY_HOST, _provider_for_host
    from agent.model_metadata import _infer_provider_from_url

    assert _infer_provider_from_url("https://api.business.fake-llm.test/v1") == NAME
    assert _infer_provider_from_url("https://catalog.fake-meta.test/inference") == NAME
    assert _provider_for_host("https://api.fake-llm.test/v1", _POOL_PROVIDER_BY_HOST) == NAME
    assert _provider_for_host("https://api.fake-llm.test/v1", _AUTH_REFRESH_PROVIDER_BY_HOST) == NAME
    # credential routing never follows a metadata-only claim (other credentials live there)
    assert _provider_for_host("https://catalog.fake-meta.test", _POOL_PROVIDER_BY_HOST) is None


def test_live_context_length_answers_after_cache_with_credentials(fake_profile):
    from agent.model_metadata import _resolve_provider_aware_context_length

    ctx = _resolve_provider_aware_context_length("m-1", "https://api.fake-llm.test/v1", "sk-x", NAME, NAME)
    assert ctx == 777_000
    assert fake_profile.calls == [("m-1", "sk-x", "https://api.fake-llm.test/v1")]


def test_profile_for_host_answers_follow_registry_changes(request):
    """Host answers are memoized for the wire hot path, but a profile registered later is seen at once."""
    host = "https://api.late-claim.test/v1"
    providers.list_providers()
    assert providers.profile_for_host(host) is None
    late = ProviderProfile(name="l4-late-claim", hostnames=("late-claim.test",))
    providers.register_provider(late)

    def _cleanup():
        providers._REGISTRY.pop(late.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    assert providers.profile_for_host(host) is late
    _cleanup()
    assert providers.profile_for_host(host) is None


class _CountingWire(ProviderProfile):
    def request_headers(self, *, base_url, model="", user_initiated=False):
        self.calls.append((base_url, model, user_initiated))
        return {"X-Count": "user"} if user_initiated else {}


def test_pure_per_request_hook_is_asked_once_per_argument_set(request):
    """``request_headers`` is pure: a hosted plugin answers once per distinct request shape, not per call."""
    from agent.provider_route_hooks import route_answer

    profile = _CountingWire(name="l4-counting-wire", hostnames=("counting-wire.test",))
    profile.calls = []
    providers.list_providers()
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    url = "https://api.counting-wire.test/v1"
    for _ in range(3):
        answer = route_answer("request_headers", provider="custom", base_url=url, model="m", user_initiated=True)
        assert answer == {"X-Count": "user"}
        answer["mutated"] = "1"  # callers own their copy; the memo is never poisoned
    assert len(profile.calls) == 1
    route_answer("request_headers", provider="custom", base_url=url, model="m", user_initiated=False)
    assert len(profile.calls) == 2


class _ClearedByAnotherThread(dict):
    """A memo another thread clears right after every write (the 512-entry reset racing a request)."""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.clear()

    def __contains__(self, key):  # a check that passes, then the entry is gone before the read
        found = super().__contains__(key)
        self.clear()
        return found


def test_pure_hook_answer_survives_a_concurrent_memo_clear(request, monkeypatch):
    from agent import provider_route_hooks

    profile = _CountingWire(name="l4-cleared-memo", hostnames=("cleared-memo.test",))
    profile.calls = []
    providers.list_providers()
    providers.register_provider(profile)

    def _cleanup():
        providers._REGISTRY.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    monkeypatch.setattr(provider_route_hooks, "_PURE_MEMO", _ClearedByAnotherThread())
    answer = provider_route_hooks.route_answer(
        "request_headers", provider="custom", base_url="https://api.cleared-memo.test/v1", model="m",
        user_initiated=True)
    assert answer == {"X-Count": "user"}
