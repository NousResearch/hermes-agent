"""Tests for the ``daily_and_idle`` session reset mode.

``daily_and_idle`` resets a session at the daily boundary (``at_hour``) ONLY
when the session has ALSO been idle for ``idle_minutes``. A session touched
after the previous daily boundary survives the rollover, so an active
conversation that ran past midnight is not wiped at 4am.

The existing ``both`` mode resets whenever EITHER trigger fires first, which
is what wiped Discord threads at 4am even when they were used minutes earlier.
"""

from datetime import datetime, timedelta

import pytest

from gateway.config import SessionResetPolicy
from gateway.session import SessionStore


def _make_store(policy: SessionResetPolicy) -> SessionStore:
    """Build a SessionStore whose reset policy resolves to ``policy``.

    Avoids loading a full GatewayConfig; ``get_reset_policy`` is the only
    policy accessor ``_is_session_expired`` uses.
    """
    store = object.__new__(SessionStore)
    store._db = None
    store._has_active_processes_fn = lambda *a, **k: False

    class _Config:
        def get_reset_policy(self, platform=None, session_type=None):
            return policy

    store.config = _Config()
    return store


def _entry(updated_at: datetime, platform="discord", chat_type="thread"):
    from gateway.session import SessionEntry

    return SessionEntry(
        session_key="k",
        session_id="s",
        created_at=updated_at,
        updated_at=updated_at,
        platform=platform,
        chat_type=chat_type,
    )


# Baseline: ``both`` resets at the daily boundary even if recently active.
def test_both_resets_at_daily_boundary_regardless_of_recent_activity():
    policy = SessionResetPolicy(mode="both", at_hour=4, idle_minutes=1440)
    store = _make_store(policy)
    # Now is 4:30am; last activity was 3:55am (35 min ago) — well within idle.
    now = datetime(2026, 8, 23, 4, 30, 0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: now)
        entry = _entry(datetime(2026, 8, 23, 3, 55, 0))
        assert store._is_session_expired(entry) is True


# The new mode: survives the daily boundary when recently active.
def test_daily_and_idle_survives_daily_boundary_when_recently_active():
    policy = SessionResetPolicy(mode="daily_and_idle", at_hour=4, idle_minutes=1440)
    store = _make_store(policy)
    now = datetime(2026, 8, 23, 4, 30, 0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: now)
        # Active at 3:55am — less than idle_minutes ago AND after the 4am
        # boundary of the prior day, so it must NOT reset.
        entry = _entry(datetime(2026, 8, 23, 3, 55, 0))
        assert store._is_session_expired(entry) is False


# The new mode: resets at the daily boundary when ALSO idle past the threshold.
def test_daily_and_idle_resets_at_daily_boundary_when_idle():
    # Now 4:30am; last activity 1:00am. With idle_minutes=60 the session is
    # idle past the window, and 1:00am is before the 4am boundary, so it resets.
    policy = SessionResetPolicy(mode="daily_and_idle", at_hour=4, idle_minutes=60)
    store = _make_store(policy)
    now = datetime(2026, 8, 23, 4, 30, 0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: now)
        entry = _entry(datetime(2026, 8, 23, 1, 0, 0))
        assert store._is_session_expired(entry) is True


# The new mode: idle but before the daily boundary does NOT reset.
def test_daily_and_idle_idle_but_before_boundary_does_not_reset():
    policy = SessionResetPolicy(mode="daily_and_idle", at_hour=4, idle_minutes=60)
    store = _make_store(policy)
    # Now is 3:00am (before the 4am boundary); last activity 1:00am (>60m ago).
    # Daily gate fails (activity 01:00 Aug 23 is after the prior 4am boundary
    # of Aug 22 04:00), so no reset even though idle_minutes is exceeded.
    now = datetime(2026, 8, 23, 3, 0, 0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: now)
        entry = _entry(datetime(2026, 8, 23, 1, 0, 0))
        assert store._is_session_expired(entry) is False


def test_daily_and_idle_resets_only_after_the_later_gate_crosses():
    """The AND result is independent of whether daily or idle becomes true first."""
    store = _make_store(SessionResetPolicy(mode="daily_and_idle", at_hour=4, idle_minutes=60))

    # Idle is already true at 03:30, but the prior day's daily boundary still
    # applies. Once 04:00 is crossed, both gates hold and the reset occurs.
    entry_idle_first = _entry(datetime(2026, 8, 23, 2, 0, 0))
    # The daily boundary is crossed at 04:00, but the idle deadline (04:30)
    # has not. The reset waits until the idle gate is crossed too.
    entry_daily_first = _entry(datetime(2026, 8, 23, 3, 30, 0))

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: datetime(2026, 8, 23, 3, 30, 0))
        assert store._is_session_expired(entry_idle_first) is False

        mp.setattr("gateway.session._now", lambda: datetime(2026, 8, 23, 4, 1, 0))
        assert store._is_session_expired(entry_idle_first) is True
        assert store._is_session_expired(entry_daily_first) is False

        mp.setattr("gateway.session._now", lambda: datetime(2026, 8, 23, 4, 31, 0))
        assert store._is_session_expired(entry_daily_first) is True


# Exact-boundary case: activity AT the boundary timestamp is NOT past it.
def test_daily_and_idle_activity_exactly_at_boundary_survives():
    policy = SessionResetPolicy(mode="daily_and_idle", at_hour=4, idle_minutes=10)
    store = _make_store(policy)
    # Now 4:30am; activity at exactly 4:00:00.000000 (the boundary). Although
    # the idle deadline has passed, updated_at is NOT < boundary, so no reset.
    now = datetime(2026, 8, 23, 4, 30, 0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: now)
        entry = _entry(datetime(2026, 8, 23, 4, 0, 0, 0))
        assert store._is_session_expired(entry) is False


# The shared boundary helper behaves at the exact-hour edge.
def test_daily_reset_boundary_helper_exact_hour():
    from gateway.session import SessionStore as _S

    # now == 4:00:00 -> today's boundary (not yesterday's)
    assert _S._daily_reset_boundary(datetime(2026, 8, 23, 4, 0, 0), 4) == datetime(2026, 8, 23, 4, 0, 0)
    # now == 3:59:59 -> yesterday's boundary
    assert _S._daily_reset_boundary(datetime(2026, 8, 23, 3, 59, 59), 4) == datetime(2026, 8, 22, 4, 0, 0)
    # now == 4:00:01 -> today's boundary
    assert _S._daily_reset_boundary(datetime(2026, 8, 23, 4, 0, 1), 4) == datetime(2026, 8, 23, 4, 0, 0)
    # at_hour == 0 (midnight)
    assert _S._daily_reset_boundary(datetime(2026, 8, 23, 0, 30, 0), 0) == datetime(2026, 8, 23, 0, 0, 0)


# Invalid mode / out-of-range at_hour are rejected by the policy loader.
def test_session_reset_policy_rejects_invalid_mode_and_hour():
    p = SessionResetPolicy.from_dict({"mode": "bogus", "at_hour": 99, "idle_minutes": 1440})
    assert p.mode == "none"  # invalid mode falls back
    assert p.at_hour == 4     # out-of-range hour falls back
    p2 = SessionResetPolicy.from_dict({"mode": "daily_and_idle", "at_hour": "not-a-number", "idle_minutes": 60})
    assert p2.mode == "daily_and_idle"
    assert p2.at_hour == 4

    for at_hour in (True, 4.0, float("inf")):
        policy = SessionResetPolicy.from_dict({"mode": "daily", "at_hour": at_hour})
        assert policy.at_hour == 4


def test_session_reset_policy_sanitizes_non_scalar_mode_and_invalid_idle_timeout():
    """Malformed YAML must not crash policy loading or make daily_and_idle daily-only."""
    for mode in ([], {}):
        policy = SessionResetPolicy.from_dict({"mode": mode, "idle_minutes": "not-a-number"})
        assert policy.mode == "none"
        assert policy.idle_minutes == 1440

    for idle_minutes in (0, -1, True, 4.5, 10**20, "not-a-number"):
        policy = SessionResetPolicy.from_dict(
            {"mode": "daily_and_idle", "idle_minutes": idle_minutes}
        )
        assert policy.idle_minutes == 1440

    assert SessionResetPolicy.from_dict({"idle_minutes": "1440"}).idle_minutes == 1440


# Both exhausted: idle & past boundary.
def test_daily_and_idle_expired_via_idle_window():
    policy = SessionResetPolicy(mode="daily_and_idle", at_hour=4, idle_minutes=1440)
    store = _make_store(policy)
    # Now 4:30am; last activity 2 days ago (past boundary AND past idle).
    now = datetime(2026, 8, 23, 4, 30, 0)
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("gateway.session._now", lambda: now)
        entry = _entry(datetime(2026, 8, 21, 9, 0, 0))
        assert store._is_session_expired(entry) is True


def test_daily_and_idle_is_distinct_from_both_in_policy():
    p = SessionResetPolicy.from_dict({"mode": "daily_and_idle", "at_hour": 4, "idle_minutes": 1440})
    assert p.mode == "daily_and_idle"
    assert p.to_dict()["mode"] == "daily_and_idle"
