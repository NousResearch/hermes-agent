"""Tests for agent.api_max_retries config surface.

Closes #11616 — make the hardcoded ``max_retries = 3`` in the agent's API
retry loop user-configurable so fallback-provider setups can fail over
faster on flaky primaries instead of burning ~3x180s on the same stall.
"""
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(api_max_retries=None):
    """Build an AIAgent with a mocked config.load_config that returns a
    config tree containing the given agent.api_max_retries (or default)."""
    cfg = {"agent": {}}
    if api_max_retries is not None:
        cfg["agent"]["api_max_retries"] = api_max_retries

    with patch("agent.process_bootstrap.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )




def _make_agent_with(agent_section):
    """Build an AIAgent whose config tree carries ``agent_section`` verbatim."""
    cfg = {"agent": dict(agent_section)}
    with patch("agent.process_bootstrap.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_max_rate_limit_wait_seconds_honors_config_and_defaults_off():
    """The rate-limit wait budget is opt-in: absent or negative means old behaviour (off)."""
    assert _make_agent_with({"max_rate_limit_wait_seconds": 120}).max_rate_limit_wait_seconds == 120.0
    assert _make_agent_with({}).max_rate_limit_wait_seconds == 0.0
    assert _make_agent_with({"max_rate_limit_wait_seconds": -5}).max_rate_limit_wait_seconds == 0.0


def test_api_max_retries_honors_config_override():
    """Setting agent.api_max_retries in config propagates to the agent."""
    agent = _make_agent(api_max_retries=1)
    assert agent._api_max_retries == 1

    agent2 = _make_agent(api_max_retries=5)
    assert agent2._api_max_retries == 5




