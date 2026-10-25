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
