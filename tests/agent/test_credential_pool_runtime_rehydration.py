"""Regression: cached agents can recover a missing credential pool after a 429.

A gateway session may keep an AIAgent built before pool routing was attached.  When
its API key belongs to the live pool, rate-limit recovery must rehydrate that pool
and rotate to the next credential rather than spending the retry budget on one key.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason


def test_second_429_rehydrates_matching_live_pool_and_rotates():
    """A cached pool-less Codex agent rotates after its one same-key retry."""
    from run_agent import AIAgent

    failed = SimpleNamespace(id="failed", runtime_api_key="key-failed", last_status=None, priority=0)
    replacement = SimpleNamespace(id="replacement", runtime_api_key="key-next", last_status=None, priority=1)
    pool = MagicMock()
    pool.provider = "openai-codex"
    pool.entries.return_value = [failed, replacement]
    pool.current.return_value = failed
    pool.mark_exhausted_and_rotate.return_value = replacement

    agent = MagicMock(spec=AIAgent)
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent.api_key = "key-failed"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.model = "gpt-6-astra"
    agent._swap_credential.return_value = True

    with patch("agent.agent_runtime_helpers.load_pool", return_value=pool):
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
            error_context={"reason": "usage_limit_reached"},
        )

    assert recovered is True
    assert retried is False
    assert agent._credential_pool is pool
    pool.mark_exhausted_and_rotate.assert_called_once_with(
        status_code=429,
        error_context={"reason": "usage_limit_reached"},
        api_key_hint="key-failed",
        credential_id="failed",
        failure_reason="rate_limit",
        model="gpt-6-astra",
    )
    agent._swap_credential.assert_called_once_with(replacement)
