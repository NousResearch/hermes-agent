"""Tests for agent.api_max_retries config surface.

Closes #11616 — make the hardcoded ``max_retries = 3`` in the agent's API
retry loop user-configurable so fallback-provider setups can fail over
faster on flaky primaries instead of burning ~3x180s on the same stall.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(api_max_retries=None):
    """Build an AIAgent with a mocked config.load_config that returns a
    config tree containing the given agent.api_max_retries (or default)."""
    cfg = {"agent": {}}
    if api_max_retries is not None:
        cfg["agent"]["api_max_retries"] = api_max_retries

    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_default_api_max_retries_is_three():
    """No config override → legacy default of 3 retries preserved."""
    agent = _make_agent()
    assert agent._api_max_retries == 3


def test_api_max_retries_honors_config_override():
    """Setting agent.api_max_retries in config propagates to the agent."""
    agent = _make_agent(api_max_retries=1)
    assert agent._api_max_retries == 1

    agent2 = _make_agent(api_max_retries=5)
    assert agent2._api_max_retries == 5


def test_api_max_retries_clamps_below_one_to_one():
    """0 or negative values would disable the retry loop entirely
    (the ``while retry_count < max_retries`` guard would never execute),
    so clamp to 1 = single attempt, no retry."""
    agent = _make_agent(api_max_retries=0)
    assert agent._api_max_retries == 1

    agent2 = _make_agent(api_max_retries=-3)
    assert agent2._api_max_retries == 1


def test_api_max_retries_falls_back_on_invalid_value():
    """Garbage values in config don't crash agent init — fall back to 3."""
    agent = _make_agent(api_max_retries="not-a-number")
    assert agent._api_max_retries == 3

    agent2 = _make_agent(api_max_retries=None)
    # None with dict.get default fires → default(3), then int(None) raises
    # TypeError → except branch sets to 3.
    assert agent2._api_max_retries == 3
