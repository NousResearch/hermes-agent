"""Behavior contract: OpenRouter provider_routing general passthrough.

Unknown keys in the ``provider_routing`` config block must be forwarded
verbatim into the request's provider preference object, while the typed
keys (sort/only/ignore/order/require_parameters/data_collection) keep
their attribute-based semantics and win on conflict.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _stub_agent():
    class StubAgent:
        providers_allowed = None
        providers_ignored = None
        providers_order = None
        provider_sort = None
        provider_require_parameters = False
        provider_data_collection = None
        provider_extra = {}

    return StubAgent()


def _extract_extra(pr):
    from gateway.run import _provider_routing_extra

    return _provider_routing_extra(pr)


def test_unknown_keys_forwarded_typed_keys_dropped():
    pr = {
        "sort": "throughput",
        "only": ["DeepInfra"],
        "zdr": True,
        "max_price": {"prompt": 0.5, "completion": 1.5},
        "quantizations": ["fp8"],
    }
    extra = _extract_extra(pr)
    assert extra == {
        "zdr": True,
        "max_price": {"prompt": 0.5, "completion": 1.5},
        "quantizations": ["fp8"],
    }


def test_build_preferences_carries_passthrough():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.provider_extra = _extract_extra({"zdr": True, "allow_fallbacks": False})
    prefs = _provider_preferences_for_agent(agent)
    assert prefs == {"zdr": True, "allow_fallbacks": False}


def test_typed_keys_win_on_conflict():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.providers_allowed = ["real-provider"]
    agent.provider_extra = {"only": ["typo-value"], "zdr": True}
    prefs = _provider_preferences_for_agent(agent)
    assert prefs["only"] == ["real-provider"]
    assert prefs["zdr"] is True


def test_non_string_keys_skipped():
    pr = {1: "bad", "": "empty", "zdr": True}
    extra = _extract_extra(pr)
    assert extra == {"zdr": True}


def test_empty_and_none_extra_noop():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.provider_extra = {}
    assert _provider_preferences_for_agent(agent) == {}

    agent.provider_extra = None
    assert _provider_preferences_for_agent(agent) == {}
