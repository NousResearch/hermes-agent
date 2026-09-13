"""``provider_routing.models.<id>`` overlays the flat OpenRouter routing for the CURRENT agent.model."""
from types import SimpleNamespace
from unittest.mock import patch

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


def test_batch_constructed_agent_gets_request_time_overlay():
    """Batch omits constructor routing; request-time models.<id> overlay still applies."""
    from hermes_cli.config import set_config_value
    from run_agent import AIAgent

    set_config_value(
        "provider_routing.models.google/gemini-flash.only",
        '["google"]',
    )

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ),
    ):
        agent = AIAgent(
            model="google/gemini-flash",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            max_iterations=10,
            save_trajectories=False,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert not agent.providers_allowed
    assert not agent.providers_order
    prefs = cch._provider_preferences_for_agent(agent)
    assert prefs.get("only") == ["google"]
    kwargs = cch._iteration_summary_chat_kwargs(
        agent,
        [{"role": "user", "content": "hi"}],
    )
    extra = kwargs.get("extra_body") or {}
    assert extra.get("provider", {}).get("only") == ["google"]
