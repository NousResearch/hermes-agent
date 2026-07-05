"""Tests for agent.empty_response_max_retries config surface.

Closes #58670 — make the hardcoded empty-response retry count (``< 3`` in
``conversation_loop.py``) user-configurable so providers with higher
empty-response rates under load can be tuned without patching source.
"""
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(empty_response_max_retries=None):
    """Build an AIAgent with a mocked config.load_config that returns a
    config tree containing the given agent.empty_response_max_retries
    (or default)."""
    cfg = {"agent": {}}
    if empty_response_max_retries is not None:
        cfg["agent"]["empty_response_max_retries"] = empty_response_max_retries

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


def test_default_empty_response_max_retries_is_three():
    """No config override → legacy default of 3 retries preserved."""
    agent = _make_agent()
    assert agent._empty_response_max_retries == 3


def test_empty_response_max_retries_honors_config_override():
    """Setting agent.empty_response_max_retries in config propagates."""
    agent = _make_agent(empty_response_max_retries=1)
    assert agent._empty_response_max_retries == 1

    agent2 = _make_agent(empty_response_max_retries=6)
    assert agent2._empty_response_max_retries == 6


def test_empty_response_max_retries_allows_zero_to_disable():
    """0 is a valid value — disable empty-response retries entirely so the
    agent fails over / surfaces "No reply" immediately."""
    agent = _make_agent(empty_response_max_retries=0)
    assert agent._empty_response_max_retries == 0


def test_empty_response_max_retries_clamps_negative_to_zero():
    """Negative values are meaningless for a retry ceiling → clamp to 0."""
    agent = _make_agent(empty_response_max_retries=-3)
    assert agent._empty_response_max_retries == 0


def test_empty_response_max_retries_falls_back_on_invalid_value():
    """Garbage values in config don't crash agent init — fall back to 3."""
    agent = _make_agent(empty_response_max_retries="not-a-number")
    assert agent._empty_response_max_retries == 3
