"""Regression tests for #61614: in-place model switches must not prune
stale fallback chains.

Gateway and dashboard sessions can keep an ``AIAgent`` alive while the profile's
``fallback_providers`` config changes.  A later model switch calls
``agent.switch_model()``, which historically pruned whatever
``agent._fallback_chain`` happened to hold in memory.  If that chain was stale,
the next 429 attempted the old fallback entry instead of the current config.
"""

from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import switch_model


def _make_agent(current_provider="zai", current_model="glm-5.2"):
    agent = MagicMock(name="Agent")
    agent.provider = current_provider
    agent.model = current_model
    agent.base_url = f"https://{current_provider}.example/v1"
    agent.api_key = f"{current_provider}-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="Client")
    agent._client_kwargs = {
        "api_key": agent.api_key,
        "base_url": agent.base_url,
    }
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._config_context_length = None
    agent._transport_cache = {}
    agent._cached_system_prompt = "cached-system-prompt"
    agent.context_compressor = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 2
    agent._fallback_model = {"provider": "openai-codex", "model": "gpt-5.5"}
    agent._fallback_chain = [
        {"provider": "openai-codex", "model": "gpt-5.5"},
    ]
    agent._credential_pool = MagicMock(provider=current_provider)
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    agent._create_openai_client = MagicMock(return_value=MagicMock(name="NewClient"))
    return agent


def test_switch_model_refreshes_fallback_chain_from_current_config_before_pruning():
    """A stale cached chain must be replaced by the current config on switch.

    Repro shape from #61614: the live session still holds
    ``openai-codex/gpt-5.5`` in memory, while config now says
    ``openai-codex/gpt-5.6 -> xai-oauth/grok-4.5 -> zai/glm-5.2``.  Switching
    away from ``zai`` should not preserve the stale gpt-5.5 entry, and because
    the freshly-read config explicitly names ``zai`` as a fallback, the old
    primary remains available as a user-configured later fallback.
    """
    agent = _make_agent()
    current_config = {
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.6", "reasoning_effort": "high"},
            {"provider": "xai-oauth", "model": "grok-4.5", "reasoning_effort": "high"},
            {"provider": "zai", "model": "glm-5.2", "reasoning_effort": "xhigh"},
        ]
    }

    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:zenmux")),
        patch("hermes_cli.config.load_config_readonly", return_value=current_config),
    ):
        switch_model(
            agent,
            new_model="anthropic/claude-fable-5-free",
            new_provider="custom:zenmux",
            api_key="zenmux-key",
            base_url="https://zenmux.ai/api/v1",
            api_mode="chat_completions",
        )

    assert [entry["model"] for entry in agent._fallback_chain] == [
        "gpt-5.6",
        "grok-4.5",
        "glm-5.2",
    ]
    assert agent._fallback_model == agent._fallback_chain[0]
    assert agent._fallback_index == 0
