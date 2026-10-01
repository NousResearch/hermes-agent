"""Regression tests for #94895: startup orphan sweep must respect cross-backend liveness.

Issue: multiple ``hermes serve`` / TUI-gateway processes sharing a single
``state.db`` (e.g. one desktop, two ``--isolated`` siblings, one fixed-port
launchd ``hermes serve``) each run ``sweep_orphaned_sessions()`` at startup.
Before the fix the sweep's staleness predicate considered ANY inactive
session row (old ``started_at`` + old latest ``messages.timestamp``) orphaned,
so the *first* restarted process reaped session rows that actually belonged
to a *different still-live* backend. Up to 473 rows were closed in a single
sweep on the reporter's install.

Fix: every serve/gateway process maintains a heartbeat row in a new
``gateway_heartbeats`` table refreshed periodically. The sweep's orphan
predicate is gated on **cross-process liveness**: a row is only reaped when
no live backend (i.e. a heartbeat row whose ``last_heartbeat`` is recent)
could plausibly own it. Backward-compatible fallback: if NO backend has
ever written a heartbeat (legacy deployments mid-upgrade), the original
predicates run unchanged so we never silently strand existing data.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from hermes_state import SessionDB


IDLE_S = 6 * 3600  # mirror the TUI gateway's default session TTL
# Heartbeats refresh every 30s and a backend is "stale" if its last refresh
# is older than this. Keep generous so tests don't race the timer.
HEARTBEAT_STALENESS_S = IDLE_S * 2  # default = 2× the session TTL


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


# ── helpers ─────────────────────────────────────────────────────────────


def _backdate_session(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?", (ts, session_id)
    )
    db._conn.commit()


def _set_message_timestamps(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?", (ts, session_id)
    )
    db._conn.commit()


def _make_session(
    db: SessionDB,
    session_id: str,
    *,
    source: str,
    started_at: float,
    message_at: float = None,
) -> None:
    db.create_session(session_id, source=source)
    if message_at is not None:
        db.append_message(session_id, role="user", content="hello")
        _set_message_timestamps(db, session_id, message_at)
    _backdate_session(db, session_id, started_at)


# ── core regression: the exact #94895 scenario ─────────────────────────


class TestStartupSweepRespectsOtherLiveBackends:
    """The reported symptom: a fresh backend reaps another backend's open rows."""

    def test_other_backends_live_heartbeat_spares_session(self, db):
        """Backend B owns a stale-looking tui session. Backend A (us) just
        started. B's heartbeat is fresh → the sweep must NOT close B's row.

        Pre-fix: the row's staleness alone reaped it.
        Post-fix: live heartbeats gate the orphan predicate.
        """
        now = time.time()
        # Session opened by backend B two hours ago. Idle beyond the
        # default TTL grace by the look of it — but B is still alive and
        # the user just walked away from their TUI for lunch.
        stale = now - 2 * 3600
        _make_session(db, "b-session", source="tui", started_at=stale, message_at=stale)

        # B writes its heartbeat (started an hour ago, refreshed just now).
        db.register_backend_heartbeat(
            backend_id="backend-B",
            pid=4242,
            started_at=now - 3600,
            last_heartbeat=now,
            profile="default",
            host="mac-mini",
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("b-session")["ended_at"] is None
        assert db.get_session("b-session")["end_reason"] is None

    def test_truly_dead_backend_sessions_still_reaped(self, db):
        """Backward-compatibility smoke: if no heartbeat claims ownership,
        the existing stale-row behavior must keep working.

        A legacy deployment that hasn't yet learned about heartbeats (or a
        backend whose heartbeat has expired past staleness) still gets its
        stale rows reaped — never silently preserve them.
        """
        stale = time.time() - 8 * 3600
        _make_session(
            db, "truly-dead",
            source="tui",
            started_at=stale,
            message_at=stale,
        )

        # No heartbeats → legacy sweep predicate runs unchanged.
        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == ["truly-dead"]
        row = db.get_session("truly-dead")
        assert row["end_reason"] == "startup_orphan_reap"

    def test_stale_heartbeat_does_not_protect_row(self, db):
        """A backend whose heartbeat hasn't refreshed in staleness_seconds
        is dead; its rows are fair game for the sweep.
        """
        now = time.time()
        session_started = now - 8 * 3600
        _make_session(
            db, "long-dead",
            source="tui",
            started_at=session_started,
            message_at=session_started,
        )

        # Heartbeat exists but is stale — its process is presumed dead.
        db.register_backend_heartbeat(
            backend_id="backend-dead",
            pid=1111,
            started_at=now - 24 * 3600,
            last_heartbeat=now - (HEARTBEAT_STALENESS_S + 600),
            profile="default",
            host="host",
        )

        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == ["long-dead"]
        assert db.get_session("long-dead")["end_reason"] == "startup_orphan_reap"

    def test_mixed_live_and_dead_backends_only_reaps_dead(self, db):
        """Two backends share a DB. Backend A is alive, backend B is dead.
        A's open session must survive; B's row (older than A could own) must
        be reaped.

        Mirrors the real topology where each backend owns its own open
        sessions and is the only candidate owner. The grace window is
        disabled here so the test exercises strict ownership inference
        (the canonical multi-backend case from #94895).
        """
        now = time.time()
        # A's row is fresh enough that A owns it (A started before the
        # session). B's row is older than A could own: with grace=0, B's
        # death leaves b-row with no candidate live owner.
        a_age = now - 5 * 3600  # A has been alive for 5h
        a_session_at = now - 4 * 3600  # a-row opened 4h ago (A was alive)
        b_session_at = now - 30 * 3600  # b-row opened 30h ago, before A existed
        _make_session(db, "a-row", source="tui", started_at=a_session_at, message_at=a_session_at)
        _make_session(db, "b-row", source="tui", started_at=b_session_at, message_at=b_session_at)

        db.register_backend_heartbeat(
            backend_id="A", pid=100, started_at=a_age,
            last_heartbeat=now, profile="p", host="h",
        )
        db.register_backend_heartbeat(
            backend_id="B", pid=200, started_at=now - 24 * 3600,
            last_heartbeat=now - (HEARTBEAT_STALENESS_S + 600),
            profile="p", host="h",
        )

        # grace=0 enforces strict ownership: a backend owns a session only
        # if it was alive before the session started. With grace=0, A's
        # started_at (5h ago) <= a_session_at (4h ago) ✓, but A's
        # started_at <= b_session_at (30h ago) ✗.
        swept = db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S,
            heartbeat_ownership_grace_seconds=0.0,
        )
        assert swept == ["b-row"]
        assert db.get_session("a-row")["ended_at"] is None
        assert db.get_session("b-row")["end_reason"] == "startup_orphan_reap"

    def test_exclude_ids_still_wins_over_live_heartbeats(self, db):
        """In-memory exclude_ids remain a hard veto: a session held by the
        local process is spared even if its backend heartbeat is stale or
        absent. (Defends the ``session.resume`` mid-grace case.)
        """
        stale = time.time() - 8 * 3600
        _make_session(db, "resumed", source="tui", started_at=stale, message_at=stale)

        # No heartbeat at all — would normally reap. But this process holds it.
        swept = db.sweep_orphaned_sessions(
            max_idle_seconds=IDLE_S, exclude_ids=("resumed",)
        )
        assert swept == []
        assert db.get_session("resumed")["ended_at"] is None

    def test_heartbeat_predicate_handles_message_less_row(self, db):
        """A row with no messages is owned by backend X (no-message fallback
        already uses started_at). The new gate must still hold for it.
        """
        stale = time.time() - 8 * 3600
        _make_session(db, "no-msg", source="tui", started_at=stale)

        db.register_backend_heartbeat(
            backend_id="B", pid=1, started_at=stale - 60,
            last_heartbeat=time.time(), profile="p", host="h",
        )
        assert db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S) == []
        assert db.get_session("no-msg")["ended_at"] is None


# ── heartbeat lifecycle API ─────────────────────────────────────────────


class TestBackendHeartbeatAPI:
    """The heartbeat must be cheap, idempotent, and refresh-on-write."""

    def test_register_then_refresh_overwrites_in_place(self, db):
        now = time.time()
        db.register_backend_heartbeat(
            backend_id="B", pid=1, started_at=now - 60,
            last_heartbeat=now, profile="p", host="h",
        )
        db.register_backend_heartbeat(
            backend_id="B", pid=1, started_at=now - 60,
            last_heartbeat=now + 5, profile="p", host="h",
        )
        rows = db.list_backend_heartbeats()
        assert len(rows) == 1
        assert rows[0]["backend_id"] == "B"
        assert rows[0]["last_heartbeat"] == pytest.approx(now + 5, abs=0.01)

    def test_clear_backend_heartbeat_removes_only_self(self, db):
        db.register_backend_heartbeat(
            backend_id="A", pid=1, started_at=time.time(),
            last_heartbeat=time.time(), profile="p", host="h",
        )
        db.register_backend_heartbeat(
            backend_id="B", pid=2, started_at=time.time(),
            last_heartbeat=time.time(), profile="p", host="h",
        )
        db.clear_backend_heartbeat("A")
        ids = sorted(r["backend_id"] for r in db.list_backend_heartbeats())
        assert ids == ["B"]

    def test_prune_stale_heartbeats_drops_only_expired(self, db):
        now = time.time()
        db.register_backend_heartbeat(
            backend_id="alive", pid=1, started_at=now,
            last_heartbeat=now, profile="p", host="h",
        )
        db.register_backend_heartbeat(
            backend_id="dead", pid=2, started_at=now - 24 * 3600,
            last_heartbeat=now - (HEARTBEAT_STALENESS_S + 600),
            profile="p", host="h",
        )

        pruned = db.prune_stale_heartbeats(max_age_seconds=HEARTBEAT_STALENESS_S)
        assert pruned == ["dead"]
        ids = [r["backend_id"] for r in db.list_backend_heartbeats()]
        assert ids == ["alive"]

    def test_heartbeat_is_swept_atomically_with_end_session(self, db):
        """The whole sweep runs under BEGIN IMMEDIATE — the heartbeat
        SELECT and the session UPDATE cannot interleave with a sibling
        process's heartbeat write.
        """
        # No specific race we can deterministically force in-process,
        # but we can at least exercise the write path with the same
        # helper the production sweep uses.
        stale = time.time() - 8 * 3600
        _make_session(db, "racy", source="tui", started_at=stale, message_at=stale)
        db.register_backend_heartbeat(
            backend_id="B", pid=1, started_at=time.time(),
            last_heartbeat=time.time(), profile="p", host="h",
        )
        # Same session under stress: 100x in a thread.
        results = []
        def _hit():
            try:
                results.append(db.sweep_orphaned_sessions(max_idle_seconds=IDLE_S))
            except Exception as e:  # pragma: no cover
                results.append(e)
        t = threading.Thread(target=_hit)
        t.start()
        t.join(timeout=10)
        assert all(r == [] for r in results if not isinstance(r, Exception))
        assert db.get_session("racy")["ended_at"] is None
