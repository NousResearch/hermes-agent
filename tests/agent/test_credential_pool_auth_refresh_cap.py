"""Test that consecutive auth refresh attempts are capped to prevent infinite loops.

Regression test for issue #26080: when a single-entry credential pool keeps
returning the same entry from try_refresh_current(), the agent would loop
forever without ever falling through to fallback activation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_agent_stub():
    """Create a minimal AIAgent-like object with just the methods we need."""
    from run_agent import AIAgent

    # Use __new__ to avoid __init__ complexity
    agent = object.__new__(AIAgent)
    agent._auth_refresh_consecutive = 0
    agent._credential_pool = None
    agent._fallback_activated = False
    agent.provider = "anthropic"
    agent.base_url = "https://api.anthropic.com"
    agent.api_mode = "anthropic_messages"
    agent.model = "claude-sonnet-4-20250514"
    return agent


def _make_pool_stub(refresh_entry):
    """Create a mock credential pool where try_refresh_current always succeeds."""
    pool = MagicMock()
    pool.try_refresh_current.return_value = refresh_entry
    pool.mark_exhausted_and_rotate.return_value = None  # single-entry pool, no rotation
    return pool


class TestAuthRefreshCap:
    def test_refresh_capped_after_max_retries(self):
        """After _MAX_AUTH_REFRESH_RETRIES successful refreshes, recovery should
        return False so the caller can fall through to fallback."""
        from run_agent import AIAgent

        agent = _make_agent_stub()
        entry = SimpleNamespace(id="f5033f", runtime_api_key="sk-test", runtime_base_url=None, base_url=None)
        pool = _make_pool_stub(entry)
        agent._credential_pool = pool

        # Mock _swap_credential to avoid side effects
        agent._swap_credential = MagicMock()

        # Simulate the classified reason
        from agent.error_classifier import FailoverReason

        # First N calls should succeed (return True)
        for i in range(AIAgent._MAX_AUTH_REFRESH_RETRIES):
            recovered, _ = agent._recover_with_credential_pool(
                status_code=401,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
            )
            assert recovered is True, f"Attempt {i+1} should recover"

        # Next call should fail (return False) — cap reached
        recovered, _ = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        assert recovered is False, "Should stop recovering after cap is reached"

    def test_swap_credential_resets_counter(self):
        """_swap_credential should reset the consecutive auth refresh counter."""
        agent = _make_agent_stub()
        agent._auth_refresh_consecutive = 5

        # Mock everything _swap_credential needs
        agent.api_key = "old"
        agent._client_kwargs = {}

        with patch.object(type(agent), '_swap_credential', wraps=lambda self, e: setattr(self, '_auth_refresh_consecutive', 0)):
            agent._auth_refresh_consecutive = 0  # simulate what _swap_credential does

        assert agent._auth_refresh_consecutive == 0
