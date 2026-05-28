"""Regression test for #26145: credential pool rotation after interrupt-resume.

When has_retried_429 is lost (user cancels between 429s), the pool should
still rotate if the current credential is already marked exhausted.
"""
from unittest.mock import MagicMock, patch

from agent.credential_pool import PooledCredential, STATUS_EXHAUSTED
from agent.error_classifier import FailoverReason


def _make_entry(idx, **overrides):
    defaults = dict(
        provider="test-provider",
        id=f"cred-{idx}",
        label=f"Credential {idx}",
        auth_type="api_key",
        priority=idx,
        source="manual",
        access_token=f"key-{idx}",
    )
    defaults.update(overrides)
    return PooledCredential(**defaults)


def _make_pool(entries):
    pool = MagicMock()
    pool.entries = entries
    pool.current.return_value = entries[0]
    return pool


def test_rotate_immediately_when_credential_already_exhausted():
    """If current credential has last_status='exhausted', rotate on first 429
    instead of retrying (Option A fix for #26145)."""
    entries = [_make_entry(0, last_status=STATUS_EXHAUSTED, last_error_code=429), _make_entry(1)]
    pool = _make_pool(entries)
    pool.mark_exhausted_and_rotate.return_value = entries[1]

    from run_agent import AIAgent
    with patch("run_agent.get_tool_definitions", return_value=[]),          patch("run_agent.check_toolset_requirements", return_value={}),          patch("run_agent.OpenAI"):
        agent = MagicMock(spec=AIAgent)
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,  # Key: False on first 429 after interrupt
            classified_reason=FailoverReason.rate_limit,
        )

    assert recovered is True
    assert retried is False
    pool.mark_exhausted_and_rotate.assert_called_once()
    agent._swap_credential.assert_called_once_with(entries[1])


def test_normal_retry_when_credential_not_exhausted():
    """When credential is active, first 429 should still retry (existing behavior)."""
    entries = [_make_entry(0, last_status=None), _make_entry(1)]
    pool = _make_pool(entries)

    from run_agent import AIAgent
    with patch("run_agent.get_tool_definitions", return_value=[]),          patch("run_agent.check_toolset_requirements", return_value={}),          patch("run_agent.OpenAI"):
        agent = MagicMock(spec=AIAgent)
        agent._credential_pool = pool
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
        )

    assert recovered is False
    assert retried is True
    pool.mark_exhausted_and_rotate.assert_not_called()


def test_rotate_on_second_429_when_not_exhausted():
    """When credential is active and this is the second 429, rotate (existing behavior)."""
    entries = [_make_entry(0, last_status=None), _make_entry(1)]
    pool = _make_pool(entries)
    pool.mark_exhausted_and_rotate.return_value = entries[1]

    from run_agent import AIAgent
    with patch("run_agent.get_tool_definitions", return_value=[]),          patch("run_agent.check_toolset_requirements", return_value={}),          patch("run_agent.OpenAI"):
        agent = MagicMock(spec=AIAgent)
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=True,  # Second 429
            classified_reason=FailoverReason.rate_limit,
        )

    assert recovered is True
    assert retried is False
    pool.mark_exhausted_and_rotate.assert_called_once()
