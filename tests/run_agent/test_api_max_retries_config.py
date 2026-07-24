"""Tests for the agent API retry configuration surface."""
from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(
    api_max_retries=None,
    retry_base_delay=None,
    retry_max_delay=None,
    retry_after_max_delay=None,
):
    cfg = {"agent": {}}
    values = {
        "api_max_retries": api_max_retries,
        "retry_base_delay": retry_base_delay,
        "retry_max_delay": retry_max_delay,
        "retry_after_max_delay": retry_after_max_delay,
    }
    cfg["agent"].update({key: value for key, value in values.items() if value is not None})

    with patch("run_agent.OpenAI"), patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_default_api_max_retries_is_three():
    assert _make_agent()._api_max_retries == 3


def test_default_api_retry_delays_preserve_legacy_values():
    agent = _make_agent()
    assert agent._retry_base_delay == 2.0
    assert agent._retry_max_delay == 60.0
    assert agent._retry_after_max_delay == 600.0


def test_api_retry_delays_honor_config_override():
    agent = _make_agent(
        retry_base_delay=5,
        retry_max_delay=300,
        retry_after_max_delay=900,
    )
    assert agent._retry_base_delay == 5.0
    assert agent._retry_max_delay == 300.0
    assert agent._retry_after_max_delay == 900.0


def test_api_retry_delays_reject_non_finite_and_non_positive_values():
    agent = _make_agent(
        retry_base_delay=float("nan"),
        retry_max_delay=0,
        retry_after_max_delay=float("inf"),
    )
    assert agent._retry_base_delay == 2.0
    assert agent._retry_max_delay == 60.0
    assert agent._retry_after_max_delay == 600.0

    boolean_agent = _make_agent(
        retry_base_delay=True,
        retry_max_delay=False,
        retry_after_max_delay=True,
    )
    assert boolean_agent._retry_base_delay == 2.0
    assert boolean_agent._retry_max_delay == 60.0
    assert boolean_agent._retry_after_max_delay == 600.0


def test_api_retry_max_delay_is_never_below_base_delay():
    agent = _make_agent(retry_base_delay=30, retry_max_delay=5)
    assert agent._retry_base_delay == 30.0
    assert agent._retry_max_delay == 30.0


def test_api_max_retries_honors_config_override():
    assert _make_agent(api_max_retries=1)._api_max_retries == 1
    assert _make_agent(api_max_retries=5)._api_max_retries == 5


def test_api_max_retries_clamps_below_one_to_one():
    assert _make_agent(api_max_retries=0)._api_max_retries == 1
    assert _make_agent(api_max_retries=-3)._api_max_retries == 1


def test_api_max_retries_falls_back_on_invalid_value():
    assert _make_agent(api_max_retries="not-a-number")._api_max_retries == 3
    assert _make_agent(api_max_retries=None)._api_max_retries == 3
