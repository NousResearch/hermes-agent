"""Regression test for #33175: switch_model() must roll back to the pre-swap
state if the client rebuild raises.

Before the fix, ``agent.model`` and ``agent.provider`` were assigned BEFORE
the client rebuild was attempted, with no try/except to restore them on
failure.  An exception during ``build_anthropic_client`` / OpenAI client
construction left the agent with the new model+provider name but the OLD
client — producing HTTP 400s like "claude-sonnet-4-6 is not supported on
openai-codex" on the next turn.

These tests exercise both branches (openai_chat_completions and
anthropic_messages) and assert that every mutated field returns to its
pre-swap value when the rebuild raises.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent_openrouter():
    """Agent on openrouter (openai-compatible) with sentinel client + kwargs."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalOpenRouterClient")
    agent._client_kwargs = {
        "api_key": "or-key-original",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def _make_agent_anthropic():
    """Agent on native anthropic with a sentinel anthropic client."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4-5"
    agent.base_url = "https://api.anthropic.com"
    agent.api_key = "sk-ant-original"
    agent.api_mode = "anthropic_messages"
    agent.client = None
    agent._client_kwargs = {}
    agent.context_compressor = None
    agent._anthropic_api_key = "sk-ant-original"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_client = MagicMock(name="OriginalAnthropicClient")
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def test_openai_client_rebuild_failure_rolls_back_to_original_state():
    """When OpenAI client construction fails, every mutated field must restore."""
    agent = _make_agent_openrouter()

    original_client = agent.client
    original_kwargs = dict(agent._client_kwargs)

    # _create_openai_client raises mid-swap (simulates bad key / network error)
    def boom(*_a, **_kw):
        raise RuntimeError("simulated client build failure")

    agent._create_openai_client = boom

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        with pytest.raises(RuntimeError, match="simulated client build failure"):
            agent.switch_model(
                new_model="openai/gpt-5",
                new_provider="openai-codex",
                api_key="codex-key-new",
                base_url="https://chatgpt.com/backend-api/codex/responses",
                api_mode="chat_completions",
            )

    # Core invariant: agent state is unchanged from before the call
    assert agent.model == "x-ai/grok-4"
    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"
    assert agent.api_mode == "chat_completions"
    assert agent.api_key == "or-key-original"
    assert agent.client is original_client
    assert agent._client_kwargs == original_kwargs


def test_anthropic_client_rebuild_failure_rolls_back_to_original_state():
    """When build_anthropic_client raises, every mutated field must restore."""
    agent = _make_agent_anthropic()

    original_anthropic_client = agent._anthropic_client
    original_anthropic_key = agent._anthropic_api_key
    original_anthropic_base = agent._anthropic_base_url

    with (
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=RuntimeError("simulated anthropic build failure"),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="sk-ant-resolved",
        ),
        patch("agent.anthropic_adapter._is_oauth_token", return_value=False),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        with pytest.raises(RuntimeError, match="simulated anthropic build failure"):
            agent.switch_model(
                new_model="claude-opus-4-6",
                new_provider="opencode-zen",
                api_key="zen-key-new",
                base_url="https://opencode.example/v1",
                api_mode="anthropic_messages",
            )

    # Anthropic-specific state restored
    assert agent._anthropic_client is original_anthropic_client
    assert agent._anthropic_api_key == original_anthropic_key
    assert agent._anthropic_base_url == original_anthropic_base

    # Core state also restored
    assert agent.model == "claude-sonnet-4-5"
    assert agent.provider == "anthropic"
    assert agent.base_url == "https://api.anthropic.com"
    assert agent.api_mode == "anthropic_messages"
    assert agent.api_key == "sk-ant-original"


def test_cross_branch_anthropic_to_openai_rebuild_failure_rolls_back():
    """Switching from anthropic_messages to chat_completions: failure must
    restore the anthropic state, not leave the agent half-converted."""
    agent = _make_agent_anthropic()

    original_anthropic_client = agent._anthropic_client

    def boom(*_a, **_kw):
        raise RuntimeError("openai client failed")

    agent._create_openai_client = boom

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        with pytest.raises(RuntimeError, match="openai client failed"):
            agent.switch_model(
                new_model="x-ai/grok-4",
                new_provider="openrouter",
                api_key="or-key-new",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )

    # Anthropic client preserved (not nulled by the openai branch)
    assert agent._anthropic_client is original_anthropic_client
    assert agent.model == "claude-sonnet-4-5"
    assert agent.provider == "anthropic"
    assert agent.api_mode == "anthropic_messages"
    assert agent.base_url == "https://api.anthropic.com"


def test_anthropic_post_build_failure_closes_new_client_before_inner_restore():
    agent = _make_agent_openrouter()
    old_openai_client = agent.client
    new_anthropic_client = MagicMock(name="NewAnthropicClient")
    agent._close_openai_client = MagicMock()

    with (
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=new_anthropic_client,
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="«redacted:sk-…»",
        ),
        patch(
            "agent.anthropic_adapter._is_oauth_token",
            side_effect=RuntimeError("oauth classification failed"),
        ),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        pytest.raises(RuntimeError, match="oauth classification failed"),
    ):
        agent.switch_model(
            new_model="claude-sonnet-4-5",
            new_provider="anthropic",
            api_key="«redacted:sk-…»",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        )

    assert agent.client is old_openai_client
    assert agent._anthropic_client is None
    new_anthropic_client.close.assert_called_once_with()
    agent._close_openai_client.assert_not_called()


def test_successful_anthropic_to_moa_closes_replaced_client():
    agent = _make_agent_anthropic()
    old_anthropic_client = agent._anthropic_client
    new_moa_client = MagicMock(name="NewMoAClient")
    agent._close_openai_client = MagicMock()

    with patch("agent.moa_loop.MoAClient", return_value=new_moa_client):
        agent.switch_model(
            new_model="default",
            new_provider="moa",
            api_key="",
            base_url="moa://local",
            api_mode="chat_completions",
        )

    assert agent.client is new_moa_client
    assert agent._anthropic_client is None
    assert agent._anthropic_api_key == ""
    assert agent._anthropic_base_url is None
    assert agent._is_anthropic_oauth is False
    old_anthropic_client.close.assert_called_once_with()
    agent._close_openai_client.assert_not_called()


def test_successful_anthropic_to_openai_closes_replaced_client():
    agent = _make_agent_anthropic()
    old_anthropic_client = agent._anthropic_client
    new_client = MagicMock(name="NewOpenAIClient")
    agent._create_openai_client = MagicMock(return_value=new_client)
    agent._close_openai_client = MagicMock()

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="x-ai/grok-4",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent.client is new_client
    assert agent._anthropic_client is None
    assert agent._anthropic_api_key == ""
    assert agent._anthropic_base_url is None
    assert agent._is_anthropic_oauth is False
    old_anthropic_client.close.assert_called_once_with()
    agent._close_openai_client.assert_not_called()


def test_successful_switch_still_works_after_rollback_refactor():
    """Sanity check: the try/except wrapper hasn't broken the happy path."""
    agent = _make_agent_openrouter()
    old_client = agent.client

    new_client = MagicMock(name="NewClient")
    agent._create_openai_client = lambda *_a, **_kw: new_client
    agent._close_openai_client = MagicMock()

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent.model == "openai/gpt-5"
    assert agent.provider == "openrouter"
    assert agent.api_key == "or-key-new"
    assert agent.client is new_client
    agent._close_openai_client.assert_called_once_with(
        old_client, reason="switch_model_commit", shared=True
    )


def test_late_finalization_failure_restores_nested_state_and_closes_new_client():
    """Failures after client construction must roll back the whole switch."""
    agent = _make_agent_openrouter()
    old_client = agent.client
    old_transport = MagicMock(name="OldTransport")
    new_transport = MagicMock(name="NewTransport")
    old_primary = {
        "model": "x-ai/grok-4",
        "client_kwargs": {"default_headers": {"X-Test": "old"}},
    }
    agent._transport_cache = {"chat_completions": old_transport}
    agent._primary_runtime = old_primary
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "safe-model", "meta": {"rank": 1}}
    ]
    agent._fallback_model = agent._fallback_chain[0]
    agent.context_compressor = SimpleNamespace(
        model="x-ai/grok-4", settings={"nested": {"rank": 1}}
    )
    agent._use_prompt_caching = True
    agent._use_native_cache_layout = False

    new_client = MagicMock(name="NewClient")
    agent._create_openai_client = MagicMock(return_value=new_client)
    agent._close_openai_client = MagicMock()

    def late_failure(**_kwargs):
        # Prove the graph snapshot survives nested in-place mutation while
        # preserving resource identity and fallback shared references.
        agent._fallback_chain[0]["meta"]["rank"] = 99
        agent._primary_runtime["client_kwargs"]["default_headers"]["X-Test"] = "mutated"
        agent.context_compressor.settings["nested"]["rank"] = 99
        agent._transport_cache = {"new-mode": new_transport}
        raise RuntimeError("late prompt-policy failure")

    agent._anthropic_prompt_cache_policy = late_failure

    with (
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        pytest.raises(RuntimeError, match="late prompt-policy failure"),
    ):
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent.model == "x-ai/grok-4"
    assert agent.provider == "openrouter"
    assert agent.api_key == "or-key-original"
    assert agent.client is old_client
    assert agent._transport_cache == {"chat_completions": old_transport}
    assert agent._primary_runtime == {
        "model": "x-ai/grok-4",
        "client_kwargs": {"default_headers": {"X-Test": "old"}},
    }
    assert agent._fallback_chain == [
        {"provider": "openrouter", "model": "safe-model", "meta": {"rank": 1}}
    ]
    assert agent._fallback_model is agent._fallback_chain[0]
    assert agent.context_compressor.model == "x-ai/grok-4"
    assert agent.context_compressor.settings == {"nested": {"rank": 1}}
    assert agent._cached_system_prompt == "cached"
    assert agent._use_prompt_caching is True
    assert agent._use_native_cache_layout is False
    agent._close_openai_client.assert_called_once_with(
        new_client, reason="switch_model_rollback", shared=True
    )
    new_transport.close.assert_called_once_with()
    old_transport.close.assert_not_called()
