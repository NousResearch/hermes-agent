"""Regression tests for #61614: in-place model switches must not prune
stale fallback chains.

Gateway and dashboard sessions can keep an ``AIAgent`` alive while the profile's
``fallback_providers`` config changes.  A later model switch calls
``agent.switch_model()``, which historically pruned whatever
``agent._fallback_chain`` happened to hold in memory.  If that chain was stale,
the next 429 attempted the old fallback entry instead of the current config.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import switch_model
from agent.chat_completion_helpers import try_activate_fallback
from hermes_cli.fallback_config import get_fallback_chain


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
    agent._fallback_config_chain = list(agent._fallback_chain)
    agent._credential_pool = MagicMock(provider=current_provider)
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    agent._create_openai_client = MagicMock(return_value=MagicMock(name="NewClient"))
    return agent


def _switch_to_zenmux(agent, *, config=None, config_error=None):
    chain_patch = (
        patch("hermes_cli.fallback_config.load_fallback_chain_strict", side_effect=config_error)
        if config_error is not None
        else patch(
            "hermes_cli.fallback_config.load_fallback_chain_strict",
            return_value=get_fallback_chain(config),
        )
    )
    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:zenmux")),
        chain_patch,
    ):
        switch_model(
            agent,
            new_model="anthropic/claude-fable-5-free",
            new_provider="custom:zenmux",
            api_key="zenmux-key",
            base_url="https://zenmux.ai/api/v1",
            api_mode="chat_completions",
        )


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

    _switch_to_zenmux(agent, config=current_config)

    assert [entry["model"] for entry in agent._fallback_chain] == [
        "gpt-5.6",
        "grok-4.5",
        "glm-5.2",
    ]
    assert agent._fallback_model == agent._fallback_chain[0]
    assert agent._fallback_index == 0


def test_switch_model_clears_cached_chain_after_successful_empty_config_read(
    tmp_path, monkeypatch
):
    """Removing fallback_providers must clear a live agent's stale chain."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    agent = _make_agent()

    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:zenmux")),
        patch("hermes_cli.config.get_config_path", return_value=config_path),
    ):
        switch_model(
            agent,
            new_model="anthropic/claude-fable-5-free",
            new_provider="custom:zenmux",
            api_key="zenmux-key",
            base_url="https://zenmux.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._fallback_chain == []
    assert agent._fallback_model is None


def test_switch_model_keeps_cached_chain_when_config_read_fails():
    """A transient read failure must preserve the last-known-good chain."""
    agent = _make_agent()
    cached_chain = list(agent._fallback_chain)
    _switch_to_zenmux(agent, config_error=OSError("mid-write"))

    assert agent._fallback_chain == cached_chain
    assert agent._fallback_model == cached_chain[0]


@pytest.mark.parametrize("config_text", ["fallback_providers: [", "- not-a-mapping\n"])
def test_switch_model_keeps_cached_chain_when_config_parse_fails(tmp_path, config_text):
    """Malformed or structurally invalid YAML must not clear cached fallback state."""
    agent = _make_agent()
    cached_chain = list(agent._fallback_chain)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:zenmux")),
        patch("hermes_cli.config.get_config_path", return_value=config_path),
    ):
        switch_model(
            agent,
            new_model="anthropic/claude-fable-5-free",
            new_provider="custom:zenmux",
            api_key="zenmux-key",
            base_url="https://zenmux.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._fallback_chain == cached_chain
    assert agent._fallback_model == cached_chain[0]


def test_switch_model_keeps_cached_chain_when_managed_config_parse_fails(
    tmp_path, monkeypatch
):
    """A broken managed overlay must not masquerade as an empty config."""
    user_config_path = tmp_path / "user-config.yaml"
    user_config_path.write_text("{}\n", encoding="utf-8")
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        "fallback_providers: [", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))

    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    agent = _make_agent()
    cached_chain = list(agent._fallback_chain)
    unavailable = {("openai-codex", "gpt-5.5", "")}
    agent._unavailable_fallback_keys = set(unavailable)

    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:zenmux")),
        patch("hermes_cli.config.get_config_path", return_value=user_config_path),
    ):
        switch_model(
            agent,
            new_model="anthropic/claude-fable-5-free",
            new_provider="custom:zenmux",
            api_key="zenmux-key",
            base_url="https://zenmux.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._fallback_chain == cached_chain
    assert agent._fallback_model == cached_chain[0]
    assert agent._unavailable_fallback_keys == unavailable


def test_switch_model_uses_valid_managed_fallback_overlay(tmp_path, monkeypatch):
    """The strict snapshot must preserve managed-scope override semantics."""
    user_config_path = tmp_path / "user-config.yaml"
    user_config_path.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: stale-user-model\n",
        encoding="utf-8",
    )
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: xai-oauth\n"
        "    model: grok-4.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))

    agent = _make_agent()
    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:zenmux")),
        patch("hermes_cli.config.get_config_path", return_value=user_config_path),
    ):
        switch_model(
            agent,
            new_model="anthropic/claude-fable-5-free",
            new_provider="custom:zenmux",
            api_key="zenmux-key",
            base_url="https://zenmux.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._fallback_chain == [
        {"provider": "xai-oauth", "model": "grok-4.5"}
    ]


def test_switch_model_clears_unavailable_memo_when_fresh_chain_changes():
    """A config edit must make previously unavailable entries retryable."""
    agent = _make_agent()
    agent._unavailable_fallback_keys = {("openai-codex", "gpt-5.5", "")}
    current_config = {
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.5"},
            {"provider": "xai-oauth", "model": "grok-4.5"},
        ]
    }

    _switch_to_zenmux(agent, config=current_config)

    assert agent._unavailable_fallback_keys == set()


def test_switch_model_keeps_unavailable_memo_when_fresh_chain_is_unchanged():
    """A no-op refresh must retain session-scoped suppression state."""
    agent = _make_agent()
    unavailable = {("openai-codex", "gpt-5.5", "")}
    agent._unavailable_fallback_keys = set(unavailable)
    current_config = {
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-5.5"},
        ]
    }

    _switch_to_zenmux(agent, config=current_config)

    assert agent._unavailable_fallback_keys == unavailable


def test_repeated_switch_with_same_config_keeps_unavailable_memo_after_pruning():
    """Operational pruning must not look like an authoritative config edit."""
    agent = _make_agent()
    current_config = {
        "fallback_providers": [
            {"provider": "custom:zenmux", "model": "zenmux-fallback"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        ]
    }
    authoritative_chain = get_fallback_chain(current_config)

    _switch_to_zenmux(agent, config=current_config)
    assert agent._fallback_chain == [authoritative_chain[1]]

    unavailable = {("openrouter", "anthropic/claude-sonnet-4.6", "")}
    agent._unavailable_fallback_keys = set(unavailable)
    with (
        patch("agent.credential_pool.load_pool", return_value=MagicMock(provider="custom:other")),
        patch(
            "hermes_cli.fallback_config.load_fallback_chain_strict",
            return_value=authoritative_chain,
        ),
    ):
        switch_model(
            agent,
            new_model="other-model",
            new_provider="custom:other",
            api_key="other-key",
            base_url="https://other.example/v1",
            api_mode="chat_completions",
        )

    assert agent._unavailable_fallback_keys == unavailable


def test_switched_agent_walks_current_chain_past_unavailable_first_entry():
    """Activation must walk the refreshed chain, not the launch-time chain."""
    agent = _make_agent()
    current_config = {
        "fallback_providers": [
            {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        ]
    }

    _switch_to_zenmux(agent, config=current_config)

    fallback_client = MagicMock()
    fallback_client.api_key = "openrouter-key"
    fallback_client.base_url = "https://openrouter.ai/api/v1"
    fallback_client._custom_headers = None
    fallback_client.default_headers = None
    agent._rate_limited_until = 0
    agent._is_azure_openai_url = MagicMock(return_value=False)
    agent._is_direct_openai_url = MagicMock(return_value=False)
    agent._provider_model_requires_responses_api = MagicMock(return_value=False)
    agent._try_activate_fallback = lambda reason=None: try_activate_fallback(agent, reason)

    with (
        patch("hermes_cli.auth.get_provider_auth_state", return_value={}),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "anthropic/claude-sonnet-4.6"),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, provider: model,
        ),
    ):
        activated = agent._try_activate_fallback(None)

    assert activated is True
    assert agent.provider == "openrouter"
    assert agent.model == "anthropic/claude-sonnet-4.6"
    assert agent._fallback_index == 2
