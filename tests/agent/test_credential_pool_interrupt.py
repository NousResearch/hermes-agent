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
    pool.entries = MagicMock(return_value=entries)
    pool.current.return_value = entries[0]
    # Must be set explicitly — MagicMock.provider returns a truthy
    # child mock, which would trigger the provider-mismatch guard.
    pool.provider = ""
    return pool


def test_rotate_immediately_when_credential_already_exhausted():
    """If current credential has last_status='exhausted', rotate on first 429
    instead of retrying (Option A fix for #26145)."""
    entries = [_make_entry(0, last_status=STATUS_EXHAUSTED, last_error_code=429), _make_entry(1)]
    pool = _make_pool(entries)
    pool.mark_exhausted_and_rotate.return_value = entries[1]

    from run_agent import AIAgent
    with patch("model_tools.get_tool_definitions", return_value=[]),          patch("model_tools.check_toolset_requirements", return_value={}),          patch("agent.process_bootstrap.OpenAI"):
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


def test_rotate_daily_free_limit_immediately_on_first_429():
    """A provider-declared daily quota is a usage-limit rotation, not a retry."""
    entries = [_make_entry(0), _make_entry(1)]
    pool = _make_pool(entries)
    pool.mark_exhausted_and_rotate.return_value = entries[1]

    from run_agent import AIAgent
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI"):
        agent = MagicMock(spec=AIAgent)
        agent._credential_pool = pool
        agent.provider = "test-provider"
        agent.model = "test-model"
        agent.base_url = "https://example.invalid"
        agent.api_key = "key-0"
        agent._credential_pool_entry_id = "cred-0"
        agent._swap_credential = MagicMock()
        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
            error_context={"message": "Daily free limit reached. Try again in 23h 26m"},
        )

    assert recovered is True
    assert retried is False
    pool.mark_exhausted_and_rotate.assert_called_once()
    agent._swap_credential.assert_called_once_with(entries[1])
