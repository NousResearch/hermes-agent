"""#104445 — a standing /goal must survive idle/daily session expiry.

Session expiry (and /stop suspend) mints a fresh ``session_id`` for the same
routing key; ``load_goal`` does a flat ``goal:<session_id>`` lookup with no
lineage walk, so the active goal is orphaned on the finalized session and never
advances again. The reset path now carries the goal onto the new session the
way compression rotation does (#33618), staleness-capped so a long-dormant
thread can't resurrect an ancient autonomous loop (#91165).

Two layers are covered:
- the ``migrate_goal_to_session`` staleness cap (``max_age_seconds``);
- the ``SessionStore._migrate_goal_across_reset`` hook that wires it into the
  auto-reset lineage.

The goals DB is the HERMES_HOME-scoped ``state.db`` (state_meta), independent of
the SessionStore's own routing DB, so the hook is exercised against a real goals
store without standing up the full gateway.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import (
    SessionEntry, SessionStore, _GOAL_RESET_MIGRATION_MAX_AGE_S, _RouteDecision,
)
from hermes_cli import goals
from hermes_cli.goals import GoalState, clear_goal, load_goal, migrate_goal_to_session, save_goal


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point the goals ``state.db`` at an isolated home and clear the cached handle.

    The goals module caches SessionDB per resolved home; pin the context-local override
    to THIS home so a leak from an earlier test in the xdist worker can't redirect the
    goals DB at a dead tmp dir (mirrors ``test_goal_resume_restart``)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()
    reset_hermes_home_override(token)


def _active_goal(*, age_seconds: float = 0.0) -> GoalState:
    """An active goal whose last activity is ``age_seconds`` in the past."""
    ts = time.time() - age_seconds
    return GoalState(goal="ship the thing", status="active", created_at=ts, last_turn_at=ts)


# ---------------------------------------------------------------------------
# migrate_goal_to_session staleness cap
# ---------------------------------------------------------------------------

class TestMigrateStalenessCap:

    def test_migrates_fresh_goal_and_archives_parent(self, hermes_home):
        save_goal("sid_old", _active_goal())
        assert migrate_goal_to_session("sid_old", "sid_new", reason="idle", max_age_seconds=3600) is True
        carried = load_goal("sid_new")
        assert carried is not None and carried.goal == "ship the thing"
        # Exactly one ACTIVE row: the parent is archived as ``cleared``, not left active.
        assert load_goal("sid_old").status == "cleared"

    def test_skips_goal_older_than_cap(self, hermes_home):
        save_goal("sid_old", _active_goal(age_seconds=48 * 3600))
        assert migrate_goal_to_session("sid_old", "sid_new", reason="idle", max_age_seconds=24 * 3600) is False
        assert load_goal("sid_new") is None
        # Left behind, untouched — not resurrected, not destroyed.
        assert load_goal("sid_old") is not None

    def test_no_cap_migrates_regardless_of_age(self, hermes_home):
        """The compression path passes no cap, so its behaviour is unchanged even for old goals."""
        save_goal("sid_old", _active_goal(age_seconds=90 * 24 * 3600))
        assert migrate_goal_to_session("sid_old", "sid_new", reason="compression") is True
        assert load_goal("sid_new") is not None

    def test_does_not_clobber_goal_already_on_child(self, hermes_home):
        save_goal("sid_old", _active_goal())
        save_goal("sid_new", GoalState(goal="already here", created_at=time.time()))
        assert migrate_goal_to_session("sid_old", "sid_new", max_age_seconds=3600) is False
        assert load_goal("sid_new").goal == "already here"


# ---------------------------------------------------------------------------
# SessionStore._migrate_goal_across_reset hook
# ---------------------------------------------------------------------------

def _store(tmp_path) -> SessionStore:
    config = GatewayConfig(default_reset_policy=SessionResetPolicy(mode="none"))
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._loaded = True
    return store


def _entry(session_id: str) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key="telegram:dm:42", session_id=session_id, created_at=now, updated_at=now,
        platform=Platform.TELEGRAM, chat_type="dm",
    )


def _reset_decision(prev_sid: str, new_sid: str, reason: str = "idle") -> _RouteDecision:
    decision = _RouteDecision()
    decision.prev_session_id = prev_sid
    decision.reset_reason = reason
    decision.entry = _entry(new_sid)
    return decision


class TestMigrateGoalAcrossResetHook:

    def test_carries_active_goal_to_reset_child(self, tmp_path, hermes_home):
        save_goal("sid_a", _active_goal())
        _store(tmp_path)._migrate_goal_across_reset(_reset_decision("sid_a", "sid_b"))
        assert load_goal("sid_b") is not None
        # Parent archived as ``cleared`` so it is no longer counted active.
        assert load_goal("sid_a").status == "cleared"

    def test_stale_goal_not_resurrected_on_reset(self, tmp_path, hermes_home):
        """A goal dormant past the 24h cap stays on the finalized session (#91165)."""
        save_goal("sid_a", _active_goal(age_seconds=_GOAL_RESET_MIGRATION_MAX_AGE_S + 3600))
        _store(tmp_path)._migrate_goal_across_reset(_reset_decision("sid_a", "sid_b"))
        assert load_goal("sid_b") is None
        assert load_goal("sid_a") is not None

    def test_noop_when_no_reset_lineage(self, tmp_path, hermes_home):
        save_goal("sid_a", _active_goal())
        decision = _RouteDecision()
        decision.prev_session_id = None
        decision.entry = _entry("sid_a")
        _store(tmp_path)._migrate_goal_across_reset(decision)
        # No prev lineage: the goal stays exactly where it is.
        assert load_goal("sid_a") is not None

    def test_noop_when_prev_equals_new(self, tmp_path, hermes_home):
        save_goal("sid_a", _active_goal())
        _store(tmp_path)._migrate_goal_across_reset(_reset_decision("sid_a", "sid_a"))
        assert load_goal("sid_a") is not None

    def test_migration_failure_never_raises(self, tmp_path, hermes_home):
        """A migration hiccup must never bubble out and block routing."""
        with patch("hermes_cli.goals.migrate_goal_to_session", side_effect=RuntimeError("boom")):
            # Must not raise.
            _store(tmp_path)._migrate_goal_across_reset(_reset_decision("sid_a", "sid_b"))
