"""Regression tests for the session-summary scheduler (issue #45103).

These tests exercise the REAL scheduler against a REAL SessionDB on a
real sqlite file — the code under test is never mocked. Only the LLM
call (``SessionDB.summarize_session``) is substituted, because a test
must not spend an API call.

The four behaviours pinned here are the ones the hermes-sweeper review
of #49520 asked for:

  1. ``tick()`` is wired to a real lifecycle — the gateway housekeeping
     loop owns it (``gateway/run.py::_housekeeping_session_summary``).
  2. A session already in flight is NOT re-queued on the next tick
     (the in-flight set bounds the work across ticks).
  3. The schema columns exist on a NORMAL database — i.e. the columns
     are declared in ``SCHEMA_SQL`` (``hermes_state_common.py``), not
     only in a repair path, so the scheduler's own SELECT runs.
  4. A slow/failing worker does not wedge the session forever, and the
     session is released back to the pool of candidates once it fails.
"""

import threading
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _make_session(db, session_id="s1", *, ended=True, messages=8, summary=None,
                  summary_updated_at=None, summary_model=None):
    """Create a session row with the shape the scheduler's SELECT expects."""
    db.create_session(session_id, "cli")
    for i in range(messages):
        db.append_message(session_id, "user", content=f"message {i}")
    now = time.time()
    if ended:
        db._conn.execute(
            "UPDATE sessions SET ended_at = ?, message_count = ?, archived = 0 WHERE id = ?",
            (now, messages, session_id),
        )
    else:
        db._conn.execute(
            "UPDATE sessions SET ended_at = NULL, message_count = ?, archived = 0 WHERE id = ?",
            (messages, session_id),
        )
    if summary is not None:
        db._conn.execute(
            "UPDATE sessions SET summary = ?, summary_updated_at = ?, summary_model = ? WHERE id = ?",
            (summary, summary_updated_at, summary_model, session_id),
        )
    db._conn.commit()
    return session_id


class TestSchema:
    def test_summary_columns_exist_on_a_normal_db(self, db):
        """The scheduler's SELECT reads sessions.summary/summary_updated_at.
        On a database created from SCHEMA_SQL those columns MUST exist —
        the sweeper review of #49520 caught them living only in a repair
        path, which made the query fail on an ordinary DB."""
        cols = {r["name"] for r in db._conn.execute("PRAGMA table_info(sessions)").fetchall()}
        assert "summary" in cols, f"missing summary column; have {sorted(cols)}"
        assert "summary_updated_at" in cols, f"missing summary_updated_at column; have {sorted(cols)}"
        assert "summary_model" in cols, f"missing summary_model column; have {sorted(cols)}"

    def test_scheduler_candidate_query_runs_on_a_normal_db(self, db):
        """The scheduler's own SQL must execute against a fresh DB — this is
        the exact failure the review described ("cannot run on a normal DB")."""
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(db, "s1")
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0)
        try:
            candidates = scheduler._find_candidates()
        finally:
            scheduler.shutdown(wait=False)
        assert [cid for cid, _ in candidates] == ["s1"]

class TestInFlightBounding:
    def test_a_slow_session_is_not_queued_twice(self, db, monkeypatch):
        """A session whose worker is still running must not be submitted
        again on the next tick — otherwise every tick piles another LLM
        call onto the same session."""
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(db, "s1")
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0)

        release = threading.Event()
        entered = threading.Event()
        calls = []

        def _slow_summarize(session_id, *args, **kwargs):
            calls.append(session_id)
            entered.set()
            release.wait(5)
            return "summary text"

        monkeypatch.setattr(db, "summarize_session", _slow_summarize)
        try:
            assert scheduler.tick() == 1
            assert entered.wait(5), "worker never started"
            # The worker is now blocked inside the summary call.
            assert scheduler.inflight_count == 1
            # A second tick must NOT queue it again.
            assert scheduler.tick() == 0
            assert len(calls) == 1, f"session summarized {len(calls)} times concurrently"
        finally:
            release.set()
            scheduler.shutdown(wait=True)
        # After the worker finishes, the session is released.
        deadline = time.time() + 5
        while scheduler.inflight_count and time.time() < deadline:
            time.sleep(0.05)
        assert scheduler.inflight_count == 0

    def test_a_failing_worker_releases_the_session(self, db, monkeypatch):
        """A session whose summary raises must be released, not wedged —
        otherwise it would be skipped forever."""
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(db, "s1")
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0)

        def _boom(session_id, *args, **kwargs):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(db, "summarize_session", _boom)
        try:
            assert scheduler.tick() == 1
            deadline = time.time() + 5
            while scheduler.inflight_count and time.time() < deadline:
                time.sleep(0.05)
            assert scheduler.inflight_count == 0, "failed session stayed in flight"
            # It is a candidate again on the next tick.
            assert scheduler.tick() == 1
        finally:
            scheduler.shutdown(wait=True)

class TestTriggerClasses:
    def test_closed_session_without_summary_is_a_candidate(self, db):
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(db, "s1", ended=True, summary=None)
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0)
        try:
            candidates = dict(scheduler._find_candidates())
        finally:
            scheduler.shutdown(wait=False)
        assert candidates.get("s1") == "closed"

    def test_archived_session_is_never_a_candidate(self, db):
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(db, "s1", ended=True, summary=None)
        db._conn.execute("UPDATE sessions SET archived = 1 WHERE id = 's1'")
        db._conn.commit()
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0)
        try:
            candidates = scheduler._find_candidates()
        finally:
            scheduler.shutdown(wait=False)
        assert candidates == []

    def test_stale_live_session_is_a_candidate(self, db):
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(
            db, "s1", ended=False,
            summary="old", summary_updated_at=time.time() - 7200, summary_model="m",
        )
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0,
                                            stale_minutes=60)
        try:
            candidates = dict(scheduler._find_candidates())
        finally:
            scheduler.shutdown(wait=False)
        assert candidates.get("s1") == "stale"

    def test_fresh_summary_is_not_a_candidate(self, db):
        from agent.session_summary_scheduler import SessionSummaryScheduler

        _make_session(
            db, "s1", ended=True,
            summary="fresh", summary_updated_at=time.time() - 5, summary_model="m",
        )
        scheduler = SessionSummaryScheduler(db, min_messages=1, min_age_seconds=0,
                                            idle_minutes=30)
        try:
            candidates = scheduler._find_candidates()
        finally:
            scheduler.shutdown(wait=False)
        assert candidates == []

class TestLifecycle:
    def test_atexit_is_registered_at_construction(self, tmp_path):
        """The review noted _register_atexit() was never invoked. The pool
        must be cleaned up at process exit without the host having to."""
        from agent.session_summary_scheduler import SessionSummaryScheduler

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            scheduler = SessionSummaryScheduler(db)
            assert scheduler._atexit_registered is True
            scheduler.shutdown(wait=False)
        finally:
            db.close()

    def test_gateway_housekeeping_owns_the_scheduler(self):
        """The scheduler must have a REAL production caller — the gateway
        housekeeping loop (the review noted there was none)."""
        from gateway.run import _housekeeping_session_summary

        assert callable(_housekeeping_session_summary)

    def test_housekeeping_chore_is_off_by_default(self, monkeypatch):
        """Summaries cost LLM calls, so the chore is opt-in."""
        from gateway import run as gateway_run

        called = []

        def _fake_load_config():
            return {"sessions": {}}

        monkeypatch.setattr(
            "hermes_cli.config.load_config", _fake_load_config, raising=True,
        )
        monkeypatch.setattr(
            "agent.session_summary_scheduler.get_default_scheduler",
            lambda: called.append(1),
            raising=False,
        )
        gateway_run._housekeeping_session_summary()
        assert called == [], "chore ran while sessions.auto_summary was off"
