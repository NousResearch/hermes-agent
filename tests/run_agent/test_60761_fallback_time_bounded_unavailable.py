"""
Regression tests for issue #60761 Bug 2 - a fallback entry that fails
once is permanently suppressed for the rest of the session.

Bug 1 from #60761 (529 overloaded never triggers fallback) is already
fixed at the post-retry fallback check in conversation_loop.py:3916.
Bug 2 is still real and is the focus of this PR.

The bug: when a fallback entry fails (or is marked unavailable for any
local reason), it is added to ``_unavailable_fallback_keys`` which is
**session-scoped and never cleared**. If gemini returns ``Server
disconnected without sending a response.`` on the first try, every
subsequent turn in that session skips gemini entirely — including
retries minutes later when the upstream has recovered.

The fix: replace the permanent set with a time-bounded cache
(``{key: monotonic_expiry}`` dict) where entries expire after a
configurable TTL (default 300s / 5 minutes). After the TTL elapses,
the fallback entry becomes eligible again.

Test approach: build a minimal agent, mark a fallback key as
unavailable via the new helper, advance monotonic time past the TTL,
and verify the key is no longer considered unavailable.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_agent() -> AIAgent:
    """Build a minimal AIAgent with fallback chain and the new
    time-bounded unavailability cache.
    """
    agent = object.__new__(AIAgent)
    agent._fallback_index = 0
    agent._fallback_chain = [
        {"provider": "gemini", "model": "gemini-3.5-flash"},
        {"provider": "deepseek", "model": "deepseek-v4-pro"},
    ]
    # The new cache lives at _fallback_unavailable_until (TTL-based dict).
    # Pre-populated empty for the tests.
    agent._fallback_unavailable_until = {}
    # Backwards-compat alias: the OLD attribute. Tests for the fix
    # should verify it's NOT used anymore.
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFallbackUnavailableTimeBounded:
    """The session-scoped suppression set is replaced by a time-bounded
    cache. Entries expire after the configured TTL so recovered
    upstreams become eligible again.
    """

    def test_helper_marks_entry_unavailable_with_default_ttl(self):
        """The fix adds ``AIAgent._mark_fallback_unavailable(provider, model, ttl_seconds=None)``
        which records the entry in the time-bounded cache.
        """
        agent = _bare_agent()
        assert hasattr(AIAgent, "_mark_fallback_unavailable"), (
            "AIAgent._mark_fallback_unavailable is missing; the #60761 "
            "Bug 2 fix replaces the permanent _unavailable_fallback_keys "
            "set with a time-bounded cache."
        )

        agent._mark_fallback_unavailable("gemini", "gemini-3.5-flash")
        # Cache now contains the entry; default TTL applied.
        assert "gemini/gemini-3.5-flash" in agent._fallback_unavailable_until
        expiry = agent._fallback_unavailable_until["gemini/gemini-3.5-flash"]
        # Expiry is in the future (within ~5 minutes by default).
        assert expiry > time.monotonic()

    def test_helper_marks_entry_with_explicit_ttl(self):
        """Caller-supplied TTL is honored (e.g. 60s for short-lived errors)."""
        agent = _bare_agent()
        agent._mark_fallback_unavailable("gemini", "gemini-3.5-flash", ttl_seconds=60.0)
        expiry = agent._fallback_unavailable_until["gemini/gemini-3.5-flash"]
        # Expiry should be within the next ~60s.
        now = time.monotonic()
        assert now + 50.0 < expiry < now + 70.0

    def test_entry_expires_after_ttl(self):
        """After the TTL elapses, the entry is no longer considered
        unavailable — the upstream gets another chance.
        """
        agent = _bare_agent()
        # Use a very short TTL.
        agent._mark_fallback_unavailable("gemini", "gemini-3.5-flash", ttl_seconds=0.05)
        # Right after marking: entry is present.
        assert "gemini/gemini-3.5-flash" in agent._fallback_unavailable_until
        # After the TTL elapses, the entry is no longer considered
        # unavailable. _is_fallback_available() (the helper that the
        # try_activate_fallback call site consults) returns True after expiry.
        time.sleep(0.1)
        assert agent._is_fallback_available("gemini", "gemini-3.5-flash") is True

    def test_helper_exists_for_availability_check(self):
        """The fix adds ``AIAgent._is_fallback_available(provider, model)``
        which returns True if the fallback entry is either not marked
        unavailable OR its TTL has elapsed.
        """
        assert hasattr(AIAgent, "_is_fallback_available"), (
            "AIAgent._is_fallback_available is missing; the #60761 "
            "Bug 2 fix uses this helper to consult the time-bounded cache."
        )

    def test_old_unavailable_keys_attribute_no_longer_used(self):
        """Regression guard: the OLD permanent set (_unavailable_fallback_keys)
        is no longer the source of truth. The new fix replaces it with the
        time-bounded cache. We don't require it to be removed (other code
        may still touch it), but the new helper should NOT consult it.
        """
        agent = _bare_agent()
        # Set the OLD attribute to "pollute" it — the new helper must
        # ignore the old set.
        agent._unavailable_fallback_keys = {"gemini/gemini-3.5-flash"}

        # The new helper, called WITHOUT marking anything, should return True
        # (no TTL-bound entry present, so it's available).
        assert agent._is_fallback_available("gemini", "gemini-3.5-flash") is True, (
            "the new time-bounded cache must be the source of truth; "
            "the old _unavailable_fallback_keys set should not be consulted "
            "by the new helper."
        )

    def test_no_unavailable_entries_means_all_available(self):
        """When the cache is empty, all fallback entries are considered
        available (the common case at session start).
        """
        agent = _bare_agent()
        assert agent._is_fallback_available("gemini", "gemini-3.5-flash") is True
        assert agent._is_fallback_available("deepseek", "deepseek-v4-pro") is True

    def test_unavailable_for_other_provider_does_not_block_target(self):
        """If gemini is marked unavailable, deepseek is unaffected."""
        agent = _bare_agent()
        agent._mark_fallback_unavailable("gemini", "gemini-3.5-flash", ttl_seconds=60.0)
        assert agent._is_fallback_available("gemini", "gemini-3.5-flash") is False
        assert agent._is_fallback_available("deepseek", "deepseek-v4-pro") is True

    def test_expired_entries_are_cleaned_on_access(self):
        """Calling _is_fallback_available on an expired entry cleans it
        from the cache so the dict doesn't grow unbounded.
        """
        agent = _bare_agent()
        agent._mark_fallback_unavailable("gemini", "gemini-3.5-flash", ttl_seconds=0.05)
        assert "gemini/gemini-3.5-flash" in agent._fallback_unavailable_until
        time.sleep(0.1)
        # Access cleans up.
        agent._is_fallback_available("gemini", "gemini-3.5-flash")
        # The expired entry should be cleaned out by the access.
        assert "gemini/gemini-3.5-flash" not in agent._fallback_unavailable_until