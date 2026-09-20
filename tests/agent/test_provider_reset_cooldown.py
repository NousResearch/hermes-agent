"""Provider-declared reset windows govern primary fallback cooldowns (#117484)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _arm_rate_limit_cooldown
from agent.turn_recovery import route_classified_error


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
def test_rate_limit_cooldown_prefers_only_valid_future_provider_resets(
    reset_at, expected_seconds,
):
    agent = SimpleNamespace(
        provider="openrouter",
        _primary_runtime={"provider": "openrouter"},
        _fallback_activated=False,
        _rate_limit_backoff_count=0,
    )
    with (
        patch("agent.fallback_cooldown.time.time", return_value=1_700_000_000),
        patch("agent.fallback_cooldown.time.monotonic", return_value=500),
    ):
        armed = _arm_rate_limit_cooldown(
            agent, FailoverReason.rate_limit, reset_at=reset_at,
        )

    assert armed == expected_seconds
    assert agent._rate_limited_until == 500 + expected_seconds


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
