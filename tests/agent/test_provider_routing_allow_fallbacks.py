"""Explicit falsy provider-routing values (e.g. ``allow_fallbacks: false``) must survive
onto the wire object built by ``_provider_preferences_for_agent``.

Regression: a truthiness filter (``if value``) dropped ``False``, silently discarding a
per-model ``allow_fallbacks: false`` ZDR pin so OpenRouter could still fall back to a
training-permitted host. The filter must drop only unset (``None``) keys.
"""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as cch


def _agent(model, **flat):
    base = dict(providers_allowed=None, providers_ignored=None, providers_order=None,
                provider_sort=None, provider_require_parameters=False, provider_data_collection=None)
    base.update(flat)
    return SimpleNamespace(model=model, **base)


@pytest.fixture
def zdr_cfg(monkeypatch):
    cfg = {"provider_routing": {"models": {
        "meta-llama/llama-3.3-70b-instruct": {
            "only": ["groq", "together"],
            "data_collection": "deny",
            "allow_fallbacks": False,
        },
    }}}
    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
    return cfg


def test_explicit_allow_fallbacks_false_survives_to_wire_object(zdr_cfg):
    prefs = cch._provider_preferences_for_agent(_agent("meta-llama/llama-3.3-70b-instruct"))
    # The whole point: an explicit False must not be filtered out.
    assert "allow_fallbacks" in prefs, "allow_fallbacks:false was dropped before reaching OpenRouter"
    assert prefs["allow_fallbacks"] is False
    # The rest of the pin travels alongside it.
    assert prefs["only"] == ["groq", "together"]
    assert prefs["data_collection"] == "deny"


def test_unset_keys_do_not_leak_null_or_empty(zdr_cfg):
    # A model with no per-model pin and everything unset must produce a clean object,
    # not one full of null/empty keys (guards against an over-broad ``return merged`` fix).
    prefs = cch._provider_preferences_for_agent(_agent("some/unpinned-model"))
    assert prefs == {}, f"unset keys leaked onto the wire object: {prefs}"
