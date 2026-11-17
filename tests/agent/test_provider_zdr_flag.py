"""``provider_routing.zdr`` must reach the OpenRouter ``provider`` object as a zero-retention pin.

``zdr`` is an OpenRouter provider-routing field (the request dies with ``404 All providers have been
ignored`` when no listed endpoint keeps zero retention), so it belongs in the provider-preference dict
that ``agent/transports/chat_completions.py`` writes to ``extra_body["provider"]`` — and only there: the
transport already gates that write on ``is_openrouter``.
"""

from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as cch


def _agent(model="deepseek/deepseek-v4.1-flash-1017", **flat):
    base = dict(
        providers_allowed=None,
        providers_ignored=None,
        providers_order=["deepinfra"],
        provider_sort="price",
        provider_require_parameters=False,
        provider_data_collection=None,
    )
    base.update(flat)
    return SimpleNamespace(model=model, **base)


@pytest.fixture
def cfg(monkeypatch):
    import hermes_cli.config as config_mod

    def _set(payload):
        monkeypatch.setattr(config_mod, "load_config_readonly", lambda: payload)

    return _set


def test_zdr_true_is_forwarded(cfg):
    cfg({"provider_routing": {"zdr": True, "order": ["deepinfra"]}})
    prefs = cch._provider_preferences_for_agent(_agent())
    assert prefs["zdr"] is True
    assert prefs["order"] == ["deepinfra"]


def test_zdr_absent_or_false_is_not_sent(cfg):
    cfg({"provider_routing": {"order": ["deepinfra"]}})
    assert "zdr" not in cch._provider_preferences_for_agent(_agent())

    cfg({"provider_routing": {"zdr": False}})
    assert "zdr" not in cch._provider_preferences_for_agent(_agent())

    cfg({})
    assert "zdr" not in cch._provider_preferences_for_agent(_agent())


def test_zdr_survives_a_non_dict_block(cfg):
    cfg({"provider_routing": None})
    assert "zdr" not in cch._provider_preferences_for_agent(_agent())


def test_zdr_is_overridable_per_model(cfg):
    cfg(
        {
            "provider_routing": {
                "zdr": True,
                "models": {"deepseek/deepseek-v4.1-flash-1017": {"zdr": False}},
            }
        }
    )
    assert "zdr" not in cch._provider_preferences_for_agent(_agent())
    # A model with no override keeps the flat pin.
    assert (
        cch._provider_preferences_for_agent(_agent("openai/gpt-6-astra"))["zdr"] is True
    )
