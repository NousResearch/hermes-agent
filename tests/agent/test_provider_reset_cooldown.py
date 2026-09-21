"""Provider-declared reset windows govern primary fallback cooldowns (#117484)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from agent.turn_recovery import route_classified_error
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


def test_eager_rate_limit_fallback_forwards_extracted_reset_time():
    reset_at = 1_900_000_000
    agent = MagicMock()
    agent.compression_enabled = True
    agent.provider = "openrouter"
    agent._fallback_index = 0
    agent._fallback_chain = [{"provider": "anthropic", "model": "claude"}]
    agent._credential_pool = None
    agent._try_activate_fallback.return_value = True
    classified = SimpleNamespace(reason=FailoverReason.rate_limit)
    retry = SimpleNamespace()
    api_error = SimpleNamespace(status_code=429)

    with (
        patch(
            "agent.conversation_loop._ra",
            return_value=SimpleNamespace(
                _pool_may_recover_from_rate_limit=lambda _pool: False,
            ),
        ),
        patch("agent.conversation_loop._arm_fallback_restart", return_value="fallback prompt"),
    ):
        verdict = route_classified_error(
            agent,
            api_error,
            classified,
            retry,
            error_msg="rate limited",
            error_context={"reset_at": reset_at},
            recovered_with_pool=False,
            base_url="https://openrouter.ai/api/v1",
            model="primary",
            messages=[],
            api_messages=[],
            system_message="system",
            active_system_prompt="system",
            conversation_history=[],
            retry_count=1,
            max_retries=3,
            compression_attempts=0,
            max_compression_attempts=2,
            api_call_count=1,
            effective_task_id=None,
        )

    assert verdict.action == "break"
    agent._try_activate_fallback.assert_called_once_with(
        reason=FailoverReason.rate_limit, reset_at=reset_at,
    )


def test_exhausted_retries_allow_fallback_before_imminent_reset(monkeypatch):
    """An imminent provider reset may defer fallback during retries, but not after exhaustion."""
    from agent.turn_api_error import settle_unrecovered_error

    agent = _agent_with_one_fallback()
    retry = SimpleNamespace(primary_recovery_attempted=True)
    api_error = SimpleNamespace(status_code=429, message="rate limited")
    classified = SimpleNamespace(
        reason=FailoverReason.rate_limit,
        retryable=True,
        should_compress=False,
        should_fallback=True,
        billing_unverified=False,
    )
    reset_at = 1_700_000_030

    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "gpt-5.5")),
        patch("hermes_cli.config.load_config", return_value={"fallback": {"min_switch_reset_seconds": 120}}),
        patch("agent.fallback_cooldown.time.time", return_value=1_700_000_000),
        patch("agent.conversation_loop._arm_fallback_restart", return_value="fallback prompt"),
    ):
        verdict = settle_unrecovered_error(
            agent,
            api_error=api_error,
            classified=classified,
            _retry=retry,
            status_code=429,
            error_msg="rate limited",
            is_context_length_error=False,
            is_rate_limited=True,
            _is_zai_coding_overload=False,
            _provider="openrouter",
            _base="https://openrouter.ai/api/v1",
            _model="primary/model",
            messages=[],
            api_messages=[],
            api_kwargs={},
            active_system_prompt="system",
            conversation_history=[],
            approx_tokens=0,
            retry_count=3,
            max_retries=3,
            compression_attempts=0,
            api_call_count=1,
            error_context={"reset_at": reset_at},
        )

    assert verdict.action == "break"
    assert agent.model == "gpt-5.5"
