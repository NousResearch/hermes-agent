"""Regression test for TUI v2 blitz bug: explicit /model --provider switch
silently fell back to the old primary provider on the next turn because the
fallback chain — seeded from config at agent __init__ — kept entries for the
provider the user just moved away from.

Reported: "switched from openrouter provider to anthropic api key via hermes
model and the tui keeps trying openrouter".
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(chain, provider="openrouter", model="x-ai/grok-4", base_url="https://openrouter.ai/api/v1"):
    agent = AIAgent.__new__(AIAgent)

    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.api_key = "or-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "or-key", "base_url": base_url}
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
    agent._create_openai_client = MagicMock()
    agent._apply_client_headers_for_base_url = MagicMock()

    return agent


def _switch_to_anthropic(agent):
    with (
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="sk-ant-xyz"),
        patch("agent.anthropic_credentials._is_oauth_token", return_value=False),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="claude-sonnet-4-5",
            new_provider="anthropic",
            api_key="sk-ant-xyz",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        )


def _switch_to_custom(
    agent,
    new_model="claude-opus-4.6",
    new_provider="custom:antigravity",
    base_url="http://cpa-host:8000/v1",
):
    with (
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        patch("agent.agent_runtime_helpers._apply_switched_provider_request_overrides", return_value=None),
    ):
        agent.switch_model(
            new_model=new_model,
            new_provider=new_provider,
            api_key="cpa-key",
            base_url=base_url,
            api_mode="chat_completions",
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


def test_manual_switch_clears_provider_fallback_provenance():
    agent = _make_agent([
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ])
    agent._provider_fallback_active = True
    agent._provider_fallback_route = ("fallback-model", "fallback-provider")

    _switch_to_anthropic(agent)

    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None




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


def test_switch_preserves_custom_provider_fallbacks_with_different_endpoints():
    """Regression test for #103788: switch_model() must not wipe fallback entries using bare
    'custom' when switching to/from custom providers on different base URLs."""
    chain = [
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8001/v1"},
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8002/v1"},
        {"provider": "custom", "model": "glm-5.3-flash", "base_url": "http://127.0.0.1:8001/v1"},
    ]
    agent = _make_agent(
        chain,
        provider="custom",
        model="gemini-3.8-flash-high",
        base_url="http://cpa-host:8000/v1",
    )

    _switch_to_custom(
        agent,
        new_model="claude-opus-4.6",
        new_provider="custom:antigravity",
        base_url="http://cpa-host:8000/v1",
    )

    assert agent._fallback_chain == chain
    assert agent._fallback_model == chain[0]


def test_switch_prunes_custom_fallback_matching_destination_endpoint():
    """A custom fallback entry targeting the destination endpoint must be pruned as redundant."""
    chain = [
        {"provider": "custom", "model": "cpa-dup", "base_url": "http://cpa-host:8000/v1"},
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8001/v1"},
    ]
    agent = _make_agent(
        chain,
        provider="openrouter",
        model="x-ai/grok-4",
        base_url="https://openrouter.ai/api/v1",
    )

    _switch_to_custom(
        agent,
        new_model="claude-opus-4.6",
        new_provider="custom:antigravity",
        base_url="http://cpa-host:8000/v1",
    )

    providers = [(entry["provider"], entry.get("base_url")) for entry in agent._fallback_chain]
    assert providers == [("custom", "http://127.0.0.1:8001/v1")]
    assert agent._fallback_model == {
        "provider": "custom",
        "model": "hy4-preview",
        "base_url": "http://127.0.0.1:8001/v1",
    }


def test_switch_prunes_custom_fallback_matching_old_endpoint():
    """A custom fallback entry targeting the rejected old primary endpoint must be pruned."""
    chain = [
        {"provider": "custom", "model": "old-backup", "base_url": "http://old-cpa:8000/v1"},
        {"provider": "custom", "model": "hy4-preview", "base_url": "http://127.0.0.1:8001/v1"},
    ]
    agent = _make_agent(
        chain,
        provider="custom",
        model="gemini-3.8-flash-high",
        base_url="http://old-cpa:8000/v1",
    )

    _switch_to_anthropic(agent)

    providers = [(entry["provider"], entry.get("base_url")) for entry in agent._fallback_chain]
    assert providers == [("custom", "http://127.0.0.1:8001/v1")]


def test_switch_logs_warning_when_prune_empties_chain(caplog):
    """Switching away from the only provider in fallback logs a warning when the chain is emptied."""
    import logging
    chain = [{"provider": "openrouter", "model": "x-ai/grok-4"}]
    agent = _make_agent(chain)

    with caplog.at_level(logging.WARNING):
        _switch_to_anthropic(agent)

    assert agent._fallback_chain == []
    assert any("pruned all 1 fallback chain entries" in r.message for r in caplog.records)


def test_backend_identity_endpoint_matching():
    """Test endpoint matching semantics across standard and custom providers using BackendIdentity."""
    from agent.backend_identity import BackendIdentity, FailureScope, should_skip_candidate

    def targets(entry, target_provider, target_base_url=""):
        cand = BackendIdentity.build(
            provider=entry.get("provider"),
            model=entry.get("model"),
            base_url=entry.get("base_url"),
        )
        target = BackendIdentity.build(
            provider=target_provider,
            base_url=target_base_url,
        )
        return should_skip_candidate(cand, target, FailureScope.ENDPOINT)

    # Standard registry providers match by provider name
    assert targets({"provider": "openrouter"}, "openrouter")
    assert not targets({"provider": "openrouter"}, "anthropic")
    assert not targets({"provider": "custom"}, "openrouter")
    assert not targets({"provider": "openrouter"}, "custom")

    # Custom providers match by normalized base_url
    assert targets(
        {"provider": "custom", "base_url": "http://127.0.0.1:8001/v1/"},
        "custom",
        "http://127.0.0.1:8001/v1",
    )
    assert not targets(
        {"provider": "custom", "base_url": "http://127.0.0.1:8001/v1"},
        "custom",
        "http://127.0.0.1:8002/v1",
    )

    # Named custom provider matches identical name
    assert targets({"provider": "custom:foo"}, "custom:foo")
    assert not targets({"provider": "custom:foo"}, "custom:bar")



