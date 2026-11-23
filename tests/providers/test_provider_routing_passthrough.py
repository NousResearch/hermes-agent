"""Behavior contract: OpenRouter provider_routing general passthrough.

Unknown keys in the ``provider_routing`` config block must be forwarded
verbatim into the request's provider preference object, while the typed
keys (sort/only/ignore/order/require_parameters/data_collection) keep
their attribute-based semantics and win on conflict. Per-model entries
(``provider_routing.models.<id>``) carry passthrough keys too and outrank
the flat block. Explicit values — falsy ones included — are sent
verbatim; ``None`` means unset and is never sent.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _stub_agent():
    class StubAgent:
        model = ""
        providers_allowed: object = None
        providers_ignored: object = None
        providers_order: object = None
        provider_sort: object = None
        provider_require_parameters: object = False
        provider_data_collection: object = None
        provider_extra: object = {}

    return StubAgent()


def _extract_extra(pr):
    from hermes_constants import provider_routing_extra

    return provider_routing_extra(pr)


def _patch_per_model(monkeypatch, per_model):
    monkeypatch.setattr("hermes_constants.resolve_per_model_provider_routing",
                        lambda model, models: per_model)


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


def test_models_key_is_typed_never_forwarded():
    pr = {"models": {"m1": {"zdr": True}}, "zdr": False}
    assert _extract_extra(pr) == {"zdr": False}


def test_build_preferences_carries_passthrough():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.provider_extra = _extract_extra({"zdr": True, "allow_fallbacks": False})
    prefs = _provider_preferences_for_agent(agent)
    assert prefs == {"zdr": True, "allow_fallbacks": False}


def test_falsy_extra_forwarded_verbatim_none_never():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.provider_extra = {"allow_fallbacks": False, "zdr": False, "max_price": {"prompt": 0}}
    prefs = _provider_preferences_for_agent(agent)
    assert prefs == {"allow_fallbacks": False, "zdr": False, "max_price": {"prompt": 0}}


def test_empty_containers_mean_unset_never_sent():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.provider_extra = {"quantizations": [], "headers": {}, "partition": "", "zdr": None}
    prefs = _provider_preferences_for_agent(agent)
    assert prefs == {}


def test_typed_keys_win_on_conflict():
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    agent = _stub_agent()
    agent.providers_allowed = ["real-provider"]
    agent.provider_extra = {"only": ["typo-value"], "zdr": True}
    prefs = _provider_preferences_for_agent(agent)
    assert prefs["only"] == ["real-provider"]
    assert prefs["zdr"] is True


def test_per_model_passthrough_outranks_flat(monkeypatch):
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    _patch_per_model(monkeypatch, {"quantizations": ["mxfp4"], "zdr": True})
    agent = _stub_agent()
    agent.provider_extra = {"quantizations": ["fp8"]}
    prefs = _provider_preferences_for_agent(agent)
    assert prefs["quantizations"] == ["mxfp4"]
    assert prefs["zdr"] is True


def test_per_model_typed_keys_still_overlay_flat(monkeypatch):
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    _patch_per_model(monkeypatch, {"sort": "latency"})
    agent = _stub_agent()
    agent.provider_sort = "throughput"
    agent.provider_extra = {"zdr": True}
    prefs = _provider_preferences_for_agent(agent)
    assert prefs["sort"] == "latency"
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
