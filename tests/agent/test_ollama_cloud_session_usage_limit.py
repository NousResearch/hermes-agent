"""Tests for Ollama Cloud "session usage limit" credential pool handling.

Covers two fixes:
1. ``_extract_retry_delay_seconds`` recognising "session usage limit" and
   returning a 30-minute TTL instead of falling through to the 1-hour default.
2. ``recover_with_credential_pool`` treating "session usage limit" as a
   usage-limit condition so the pool rotates on the first 429 instead of
   retrying the same credential 3 times.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.credential_pool import PooledCredential
from agent.error_classifier import FailoverReason


# ---------------------------------------------------------------------------
# Helpers (match conventions from test_credential_pool_interrupt.py)
# ---------------------------------------------------------------------------

def _make_entry(idx, **overrides):
    defaults = dict(
        provider="ollama-cloud",
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
    # Must be set explicitly — MagicMock.provider returns a truthy
    # child mock, which would trigger the provider-mismatch guard.
    pool.provider = ""
    return pool


# ---------------------------------------------------------------------------
# _extract_retry_delay_seconds
# ---------------------------------------------------------------------------

def test_session_usage_limit_returns_30_minutes():
    from agent.credential_pool import _extract_retry_delay_seconds

    msg = "you (user) have reached your session usage limit, upgrade for higher limits"
    assert _extract_retry_delay_seconds(msg) == 30 * 60


def test_session_usage_limit_case_insensitive():
    from agent.credential_pool import _extract_retry_delay_seconds

    msg = "Session Usage Limit reached"
    assert _extract_retry_delay_seconds(msg) == 30 * 60


def test_session_usage_limit_does_not_shadow_explicit_reset_time():
    """If the message also contains an explicit 'resets in Nmin', that wins."""
    from agent.credential_pool import _extract_retry_delay_seconds

    msg = "session usage limit. Resets in 5min"
    assert _extract_retry_delay_seconds(msg) == 5 * 60


def test_session_usage_limit_does_not_shadow_hr_min_format():
    """If the message also contains 'resets in Nhr Mmin', that wins."""
    from agent.credential_pool import _extract_retry_delay_seconds

    msg = "session usage limit. Resets in 2hr 15min"
    assert _extract_retry_delay_seconds(msg) == 2 * 3600 + 15 * 60


def test_non_session_usage_limit_returns_none():
    from agent.credential_pool import _extract_retry_delay_seconds

    assert _extract_retry_delay_seconds("rate limited, try again later") is None
    assert _extract_retry_delay_seconds("") is None


# ---------------------------------------------------------------------------
# recover_with_credential_pool — usage_limit_reached detection
# ---------------------------------------------------------------------------

def test_session_usage_limit_triggers_pool_rotation_on_first_429():
    """On the first 429 with 'session usage limit', the pool should rotate
    immediately instead of waiting for has_retried_429=True."""
    entries = [_make_entry(0), _make_entry(1)]
    pool = _make_pool(entries)
    pool.mark_exhausted_and_rotate.return_value = entries[1]

    from run_agent import AIAgent
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        agent = MagicMock(spec=AIAgent)
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()

        error_context = {
            "reason": "",
            "message": "you (user) have reached your session usage limit, "
                       "upgrade for higher limits",
        }

        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=None,
            error_context=error_context,
        )

    assert recovered is True
    assert retried is False
    pool.mark_exhausted_and_rotate.assert_called_once()
    agent._swap_credential.assert_called_once_with(entries[1])


def test_plain_rate_limit_without_usage_limit_waits_for_retry():
    """A normal 429 without 'session usage limit' should NOT rotate on the
    first 429 — it should retry first (has_retried_429=False → retried=True)."""
    entries = [_make_entry(0), _make_entry(1)]
    pool = _make_pool(entries)

    from run_agent import AIAgent
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI"):
        agent = MagicMock(spec=AIAgent)
        agent._credential_pool = pool

        error_context = {
            "reason": "rate_limit",
            "message": "Too many requests, try again later",
        }

        recovered, retried = AIAgent._recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=None,
            error_context=error_context,
        )

    assert recovered is False
    assert retried is True
    pool.mark_exhausted_and_rotate.assert_not_called()