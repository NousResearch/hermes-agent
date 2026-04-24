"""Regression test for TUI v2 blitz bug: explicit /model --provider switch
silently fell back to the old primary provider on the next turn because the
fallback chain — seeded from config at agent __init__ — kept entries for the
provider the user just moved away from.

Reported: "switched from openrouter provider to anthropic api key via hermes
model and the tui keeps trying openrouter".
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(chain):
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "or-key", "base_url": "https://openrouter.ai/api/v1"}
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = list(chain)
    agent._fallback_model = chain[0] if chain else None

    return agent


def _switch_to_anthropic(agent):
    with (
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-xyz"),
        patch("agent.anthropic_adapter._is_oauth_token", return_value=False),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="claude-sonnet-4-5",
            new_provider="anthropic",
            api_key="sk-ant-xyz",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        )


def test_switch_drops_old_primary_from_fallback_chain():
    agent = _make_agent([
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ])

    _switch_to_anthropic(agent)

    providers = [entry["provider"] for entry in agent._fallback_chain]

    assert "openrouter" not in providers, "old primary must be pruned"
    assert "anthropic" not in providers, "new primary is redundant in the chain"
    assert providers == ["nous"]
    assert agent._fallback_model == {"provider": "nous", "model": "hermes-4"}


def test_switch_with_empty_chain_stays_empty():
    agent = _make_agent([])

    _switch_to_anthropic(agent)

    assert agent._fallback_chain == []
    assert agent._fallback_model is None


def test_switch_initializes_missing_fallback_attrs():
    agent = _make_agent([])
    del agent._fallback_chain
    del agent._fallback_model

    _switch_to_anthropic(agent)

    assert agent._fallback_chain == []
    assert agent._fallback_model is None


def test_switch_within_same_provider_preserves_chain():
    chain = [{"provider": "openrouter", "model": "x-ai/grok-4"}]
    agent = _make_agent(chain)

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )

    assert agent._fallback_chain == chain
