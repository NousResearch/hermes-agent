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




