"""Regression test for TUI v2 blitz bug: explicit /model --provider switch
silently fell back to the old primary provider on the next turn because the
fallback chain — seeded from config at agent __init__ — kept entries for the
provider the user just moved away from.

Reported: "switched from openrouter provider to anthropic api key via hermes
model and the tui keeps trying openrouter".
"""

from types import SimpleNamespace
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


def _switch_to_anthropic(agent, *, prune_fallback_chain=True):
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
            prune_fallback_chain=prune_fallback_chain,
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


def test_switch_can_preserve_fallback_chain_for_turn_scoped_routes():
    chain = [
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ]
    agent = _make_agent(chain)

    _switch_to_anthropic(agent, prune_fallback_chain=False)

    assert getattr(agent, "_fallback_chain") == chain
    assert getattr(agent, "_fallback_model") == {
        "provider": "openrouter",
        "model": "x-ai/grok-4",
    }


def test_pre_model_route_uses_turn_scoped_runtime_activation(monkeypatch):
    agent = _make_agent([
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ])
    setattr(agent, "session_id", "sid")
    setattr(agent, "platform", "cli")
    setattr(agent, "_user_id", "")
    setattr(agent, "_chat_id", "")
    setattr(agent, "_chat_name", "")
    setattr(agent, "_chat_type", "")
    setattr(agent, "_thread_id", "")
    setattr(agent, "_gateway_session_key", "")
    setattr(agent, "_content_has_image_parts", lambda content: False)

    def _fake_activate_runtime(_agent, **kwargs):
        agent.model = kwargs["new_model"]
        agent.provider = kwargs["new_provider"]
        agent.api_key = kwargs["api_key"]
        agent.base_url = kwargs["base_url"]
        agent.api_mode = kwargs["api_mode"]

    activate_runtime = MagicMock(side_effect=_fake_activate_runtime)
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.activate_model_runtime",
        activate_runtime,
    )

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: [
            {
                "model": "claude-sonnet-4-5",
                "provider": "anthropic",
                "reason": "large context",
            }
        ],
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"providers": {}})
    monkeypatch.setattr(
        "hermes_cli.config.get_compatible_custom_providers",
        lambda _config: {},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            new_model="claude-sonnet-4-5",
            target_provider="anthropic",
            api_key="sk-ant-xyz",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        ),
    )
    runtime_main = {}
    monkeypatch.setattr(
        "agent.auxiliary_client.set_runtime_main",
        lambda provider, model, **kwargs: runtime_main.update(
            {"provider": provider, "model": model, **kwargs}
        ),
    )

    agent._apply_pre_model_route_hook(
        user_message="review this pr",
        conversation_history=[],
        is_first_turn=False,
    )

    activate_runtime.assert_called_once_with(
        agent,
        new_model="claude-sonnet-4-5",
        new_provider="anthropic",
        api_key="sk-ant-xyz",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        update_primary_runtime=False,
        prune_fallback_chain=False,
        persist_billing_route=False,
    )
    assert agent._pre_model_route_switched_this_turn is True
    assert agent._primary_runtime == {}
    assert agent._pre_model_route_restore_state["cached_system_prompt"] == "cached"
    assert agent._pre_model_route_restore_state["fallback_chain"] == [
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ]
    assert runtime_main["provider"] == "anthropic"
    assert runtime_main["model"] == "claude-sonnet-4-5"


def test_pre_model_route_without_change_leaves_switch_flag_false(monkeypatch):
    agent = _make_agent([])
    setattr(agent, "session_id", "sid")
    setattr(agent, "platform", "cli")
    setattr(agent, "_user_id", "")
    setattr(agent, "_chat_id", "")
    setattr(agent, "_chat_name", "")
    setattr(agent, "_chat_type", "")
    setattr(agent, "_thread_id", "")
    setattr(agent, "_gateway_session_key", "")
    agent.switch_model = MagicMock()

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])

    agent._apply_pre_model_route_hook(
        user_message="hello",
        conversation_history=[],
        is_first_turn=False,
    )

    agent.switch_model.assert_not_called()
    assert agent._pre_model_route_switched_this_turn is False
