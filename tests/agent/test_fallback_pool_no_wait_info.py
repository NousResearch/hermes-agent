"""next_available_at() returning None must not read as "credential pool exhausted".

#140568: a Vertex service-account key file persists as an api_key row with no
runtime token (the token is minted from the file), so the availability scan
never counts it and next_available_at() has nothing to wait for. The fallback
prefilter read that None as an unbounded cooldown and skipped every
same-provider fallback, sending a transient primary 429 straight to the user.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock


def _pool(statuses, *, until=None):
    """A pool with no available entry: entries carry *statuses*, next recovery at *until*."""
    pool = MagicMock()
    pool.provider = "vertex"
    pool.has_credentials.return_value = True
    pool.has_available.return_value = False
    pool.next_available_at.return_value = until
    pool.entries.return_value = [SimpleNamespace(last_status=s) for s in statuses]
    return pool


def _exhausted(pool):
    from agent.chat_completion_helpers import _candidate_pool_exhausted

    return _candidate_pool_exhausted(
        SimpleNamespace(_credential_pool=pool), "vertex", "google/gemini-4-argon"
    )


def test_no_wait_info_with_a_live_entry_is_not_exhausted():
    """The #140568 repro: an SA-file row never counted as available, no cooldown anywhere."""
    assert _exhausted(_pool([None])) is False


def test_no_wait_info_with_dead_and_live_entries_is_not_exhausted():
    """A revoked sibling does not bench a live entry that the scan just cannot see."""
    assert _exhausted(_pool(["dead", None])) is False


def test_all_dead_pool_is_skipped():
    """DEAD never re-enters rotation — falling back to it only fails the turn again."""
    assert _exhausted(_pool(["dead", "dead"])) is True


def test_long_cooldown_is_still_skipped():
    """#89401 regression anchor: benched past the 600s Retry-After cap stays skipped."""
    assert _exhausted(_pool(["exhausted"], until=time.time() + 30995)) is True


def test_short_throttle_still_gets_its_chance():
    assert _exhausted(_pool(["exhausted"], until=time.time() + 30)) is False
