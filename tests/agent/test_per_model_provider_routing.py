"""``provider_routing.models.<id>`` overlays the flat OpenRouter routing for the CURRENT agent.model."""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as cch


def _agent(model, **flat):
    base = dict(providers_allowed=None, providers_ignored=None, providers_order=None, provider_sort="price",
                provider_require_parameters=False, provider_data_collection=None)
    base.update(flat)
    return SimpleNamespace(model=model, **base)


@pytest.fixture
def routing_cfg(monkeypatch):
    cfg = {"provider_routing": {"sort": "price", "models": {
        "openai/gpt-6-astra": {"only": ["openai"]},
        "anthropic/claude-fable-5.1": {"only": ["anthropic"], "sort": "throughput"},
    }}}
    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
    return cfg


def test_per_model_entry_overlays_flat_routing_for_that_model_only(routing_cfg):
    assert cch._provider_preferences_for_agent(_agent("openai/gpt-6-astra")) == {"only": ["openai"], "sort": "price"}
    # A per-model key wins over the flat one; unset keys fall through.
    assert cch._provider_preferences_for_agent(_agent("anthropic/claude-fable-5.1")) == {
        "only": ["anthropic"], "sort": "throughput"}
    # Unlisted model keeps the flat behaviour; no pin leaks across models.
    assert cch._provider_preferences_for_agent(_agent("moonshotai/kimi-k2.6")) == {"sort": "price"}


def test_per_model_match_is_spelling_tolerant_and_follows_model_switch(routing_cfg):
    agent = _agent("openrouter/openai/gpt-6-astra", providers_allowed=["together"])
    assert cch._provider_preferences_for_agent(agent)["only"] == ["openai"]
    agent.model = "claude-fable-5-1"
    assert cch._provider_preferences_for_agent(agent)["only"] == ["anthropic"]


def test_zdr_flag_is_read_from_config_and_can_be_pinned_per_model(monkeypatch):
    cfg = {"provider_routing": {"zdr": True, "sort": "price",
                                "models": {"openai/gpt-6-astra": {"zdr": True, "only": ["openai"]}}}}
    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)

    # Flat config flag reaches every model …
    assert cch._provider_preferences_for_agent(_agent("moonshotai/kimi-k2.6"))["zdr"] is True
    # … and a per-model entry still overlays it alongside its own pins.
    pinned = cch._provider_preferences_for_agent(_agent("openai/gpt-6-astra"))
    assert pinned["zdr"] is True and pinned["only"] == ["openai"]


def test_zdr_is_omitted_when_not_configured(routing_cfg):
    """Absent/false zdr must not appear in the payload — OpenRouter treats a present
    key as an explicit policy demand, so a stray false would change routing."""
    assert "zdr" not in cch._provider_preferences_for_agent(_agent("moonshotai/kimi-k2.6"))
    routing_cfg["provider_routing"]["zdr"] = False
    assert "zdr" not in cch._provider_preferences_for_agent(_agent("moonshotai/kimi-k2.6"))

