from agent.chat_completion_helpers import (
    _provider_preferences_for_agent,
    _validated_openrouter_provider_sort,
)


def _agent(provider="openrouter"):
    return type(
        "Agent",
        (),
        {
            "provider": provider,
            "providers_allowed": None,
            "providers_ignored": None,
            "providers_order": None,
            "provider_sort": None,
            "provider_require_parameters": False,
            "provider_data_collection": None,
            "_is_openrouter_url": lambda self: False,
        },
    )()


def test_provider_preferences_enable_local_openrouter_zdr(monkeypatch):
    monkeypatch.setenv("HERMES_OPENROUTER_ZDR", "1")
    assert _provider_preferences_for_agent(_agent()) == {"zdr": True}


def test_provider_preferences_do_not_apply_local_zdr_to_other_providers(monkeypatch):
    monkeypatch.setenv("HERMES_OPENROUTER_ZDR", "1")
    assert _provider_preferences_for_agent(_agent("anthropic")) == {}


def test_provider_preferences_omit_local_zdr_when_disabled(monkeypatch):
    monkeypatch.delenv("HERMES_OPENROUTER_ZDR", raising=False)
    assert _provider_preferences_for_agent(_agent()) == {}


def test_validated_openrouter_provider_sort_accepts_valid_values():
    assert _validated_openrouter_provider_sort("price") == "price"
    assert _validated_openrouter_provider_sort(" latency ") == "latency"
    assert _validated_openrouter_provider_sort("THROUGHPUT") == "throughput"


def test_validated_openrouter_provider_sort_rejects_invalid_values():
    assert _validated_openrouter_provider_sort("intelligence") is None
    assert _validated_openrouter_provider_sort("") is None
    assert _validated_openrouter_provider_sort(None) is None
