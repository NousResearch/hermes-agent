"""MoA reference slots ask a provider's ``request_headers`` through the same route as the main loop:
memoized, and a failing hook keeps the agent's session on the generic answer."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def slot_profile(request, monkeypatch):
    import providers
    from agent import provider_route_hooks
    from providers.base import ProviderProfile

    state = {"calls": 0, "fail": False}

    class SlotProfile(ProviderProfile):
        def request_headers(self, *, base_url, model="", user_initiated=False):
            state["calls"] += 1
            if state["fail"]:
                raise RuntimeError("plugin bug")
            return {"X-Slot": "user"} if user_initiated else {}

    monkeypatch.setattr(provider_route_hooks, "_WARNED", set(), raising=False)
    providers.list_providers()
    providers.register_provider(SlotProfile(name="moaslot", base_url="https://api.moaslot.test/v1"))

    def _cleanup():
        providers._REGISTRY.pop("moaslot", None)
        providers._SOURCES.pop("moaslot", None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    return state


def test_slot_headers_are_memoized_per_request_shape(slot_profile):
    from agent.moa_loop import _slot_request_headers

    runtime = {"provider": "moaslot", "model": "m"}
    for _ in range(3):
        assert _slot_request_headers(runtime, SimpleNamespace()) == {"X-Slot": "user"}
    assert slot_profile["calls"] == 1


def test_failing_slot_hook_stays_generic_for_the_agent_session(slot_profile):
    from agent.moa_loop import _slot_request_headers

    runtime = {"provider": "moaslot", "model": "m"}
    agent = SimpleNamespace()
    slot_profile["fail"] = True
    assert _slot_request_headers(runtime, agent) is None
    slot_profile["fail"] = False
    assert _slot_request_headers(runtime, agent) is None  # no flip mid-session
    assert _slot_request_headers(runtime, SimpleNamespace()) == {"X-Slot": "user"}  # a new session asks again
