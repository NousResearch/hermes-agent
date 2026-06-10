"""Regression coverage for credential-pool recovery on secondary agents.

Background/secondary AIAgent paths can have a runtime API key from a pooled
credential but miss the pool object. Recovery should attach the matching pool
entry on hard quota errors instead of retrying the same credential three times.
"""
from unittest.mock import MagicMock, patch

from agent.credential_pool import PooledCredential
from agent.error_classifier import FailoverReason


def _entry(idx: int, token: str) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=f"cred-{idx}",
        label=f"Credential {idx}",
        auth_type="oauth",
        priority=idx,
        source="manual:test",
        access_token=token,
    )


def test_recovery_lazy_attaches_matching_pool_when_agent_pool_missing():
    first = _entry(0, "token-current")
    second = _entry(1, "token-next")
    pool = MagicMock()
    pool.provider = "openai-codex"
    pool.has_credentials.return_value = True
    pool.entries.return_value = [first, second]
    pool.current.return_value = None
    pool.mark_exhausted_and_rotate.return_value = second

    agent = MagicMock()
    agent._credential_pool = None
    agent.provider = "openai-codex"
    agent.api_key = "token-current"
    agent._provider_source = "hermes-auth-store"
    agent._swap_credential = MagicMock()

    from run_agent import AIAgent

    with patch("agent.credential_pool.load_pool", return_value=pool):
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
            error_context={
                "reason": "usage_limit_reached",
                "message": "The usage limit has been reached",
            },
        )

    assert recovered is True
    assert retried is False
    assert agent._credential_pool is pool
    pool.mark_exhausted_and_rotate.assert_called_once_with(
        status_code=429,
        error_context={
            "reason": "usage_limit_reached",
            "message": "The usage limit has been reached",
        },
        api_key_hint="token-current",
    )
    agent._swap_credential.assert_called_once_with(second)


def test_recovery_does_not_attach_pool_when_runtime_key_does_not_match():
    pool = MagicMock()
    pool.provider = "openai-codex"
    pool.has_credentials.return_value = True
    pool.entries.return_value = [_entry(0, "token-other")]

    agent = MagicMock()
    agent._credential_pool = None
    agent.provider = "openai-codex"
    agent.api_key = "explicit-token"
    agent._provider_source = "explicit"

    from run_agent import AIAgent

    with patch("agent.credential_pool.load_pool", return_value=pool):
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
        )

    assert recovered is False
    assert retried is False
    assert agent._credential_pool is None
    pool.mark_exhausted_and_rotate.assert_not_called()
