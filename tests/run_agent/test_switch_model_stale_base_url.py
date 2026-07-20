"""Regression tests for #47828: switch_model must not pair a new provider
label with the previous provider's base_url when the resolver returns no
new base_url for a genuine provider change.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


def _make_agent_with_compressor(provider="copilot", base_url="https://api.githubcopilot.com") -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    agent.model = "claude-opus-4.8"
    agent.provider = provider
    agent.base_url = base_url
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = None

    compressor = ContextCompressor(
        model=agent.model,
        threshold_percent=0.50,
        base_url=base_url,
        api_key="sk-primary",
        provider=provider,
        quiet_mode=True,
        config_context_length=None,
    )
    agent.context_compressor = compressor
    agent._primary_runtime = {}

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_rejects_stale_base_url_on_provider_change(mock_ctx_len):
    """A provider change with no resolved base_url must fail loud instead of
    silently keeping the previous provider's endpoint (#47828)."""
    agent = _make_agent_with_compressor(provider="copilot", base_url="https://api.githubcopilot.com")

    with pytest.raises(ValueError, match="no base_url resolved"):
        agent.switch_model("MiniMax-M3", "custom:minimax", api_key="sk-minimax", base_url="")

    # Rollback must leave the agent fully on the old (provider, base_url) pair —
    # not a mismatched new-model/old-endpoint hybrid.
    assert agent.provider == "copilot"
    assert agent.base_url == "https://api.githubcopilot.com"
    assert agent.model == "claude-opus-4.8"


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_allows_empty_base_url_for_same_provider(mock_ctx_len):
    """Re-selecting the SAME provider (e.g. a credential-only refresh) with no
    new base_url must keep the current URL — this is not a provider change."""
    agent = _make_agent_with_compressor(provider="openrouter", base_url="https://openrouter.ai/api/v1")

    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="")

    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"
    assert agent.model == "new-model"


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_applies_new_base_url_on_provider_change(mock_ctx_len):
    """The normal, resolved-correctly path must still work: new provider +
    new base_url is applied as-is."""
    agent = _make_agent_with_compressor(provider="copilot", base_url="https://api.githubcopilot.com")

    agent.switch_model(
        "MiniMax-M3", "custom:minimax", api_key="sk-minimax", base_url="https://api.minimax.io/v1"
    )

    assert agent.provider == "custom:minimax"
    assert agent.base_url == "https://api.minimax.io/v1"
    assert agent.model == "MiniMax-M3"
    # _primary_runtime must snapshot the coherent pair so it survives every
    # subsequent restore_primary_runtime() call across turns.
    assert agent._primary_runtime["provider"] == "custom:minimax"
    assert agent._primary_runtime["base_url"] == "https://api.minimax.io/v1"


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_updates_and_snapshots_responses_transport(mock_ctx_len):
    agent = _make_agent_with_compressor(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    agent.switch_model(
        "gpt-5.5",
        "openai-codex",
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        responses_transport="websocket-cached",
    )

    assert agent.responses_transport == "websocket-cached"
    assert agent._primary_runtime["responses_transport"] == "websocket-cached"


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_same_codex_provider_switch_preserves_unspecified_responses_transport(mock_ctx_len):
    agent = _make_agent_with_compressor(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    agent.responses_transport = "websocket-cached"

    agent.switch_model(
        "gpt-5.5",
        "openai-codex",
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
    )

    assert agent.responses_transport == "websocket-cached"
