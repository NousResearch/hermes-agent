"""Provider-declared reset windows govern primary fallback cooldowns (#117484)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent


def _agent_with_one_fallback():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", provider="openrouter", base_url="https://openrouter.ai/api/v1",
            model="primary/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "openai-codex", "model": "gpt-5.5"}],
        )
    agent.client = None
    return agent


def _fallback_client():
    client = SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex", api_key="fb-key")
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **k: None))
    return client


@pytest.mark.parametrize(
    ("reset_at", "expected_seconds"),
    [
        (1_700_000_090.2, 91),
        (1_700_274_291, 274_291),
        (None, 60),
        ("not-a-timestamp", 60),
        (1_699_999_999, 60),
    ],
)
def test_fallback_switch_benches_primary_until_valid_future_provider_reset(reset_at, expected_seconds):
    """Drives the production entry (agent._try_activate_fallback -> try_activate_fallback):
    the reset_at kwarg must reach the cooldown arming, not just the helper."""
    agent = _agent_with_one_fallback()
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "gpt-5.5")),
        patch("agent.fallback_cooldown.time.time", return_value=1_700_000_000),
        patch("agent.fallback_cooldown.time.monotonic", return_value=500),
    ):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=reset_at) is True

    assert agent.model == "gpt-5.5"
    assert agent._rate_limited_until == 500 + expected_seconds


@pytest.mark.parametrize(("threshold", "switched"), [(0, True), (120, False)])
def test_min_switch_reset_seconds_keeps_primary_when_reset_is_imminent(threshold, switched):
    """Opt-in fallback.min_switch_reset_seconds (#117484): a reset 30 s out is below a 120 s
    threshold, so the turn stays on the primary and no cooldown is armed; 0 (default) switches."""
    agent = _agent_with_one_fallback()
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "gpt-5.5")),
        patch("hermes_cli.config.load_config", return_value={"fallback": {"min_switch_reset_seconds": threshold}}),
        patch("agent.fallback_cooldown.time.time", return_value=1_700_000_000),
    ):
        activated = agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=1_700_000_030)

    assert activated is switched
    assert (agent.model == "gpt-5.5") is switched
    assert (getattr(agent, "_rate_limited_until", None) is not None) is switched


