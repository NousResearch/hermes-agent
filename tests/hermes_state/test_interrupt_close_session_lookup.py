"""Tests for SessionDB.most_recent_interrupt_close_session — the query behind the
CLI launch banner that nudges `hermes chat -c` when a prior turn was cut off
mid-flight (restart / reboot / terminal-close)."""
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    yield d
    d.close()


def _seed_interrupted(db, sid, *, ts):
    db.create_session(sid, source="cli", model="test-model")
    db.append_message(sid, role="user", content="do a big task", timestamp=ts - 1)
    rid = db.append_message(sid, role="assistant", content="partial...", timestamp=ts)
    db.update_message_finish_reason(sid, rid, "interrupt_close")
    return rid


def test_detects_session_whose_last_message_is_interrupt_close(db):
    now = time.time()
    _seed_interrupted(db, "sessA", ts=now)
    hit = db.most_recent_interrupt_close_session()
    assert hit is not None
    assert hit["id"] == "sessA"


def test_skips_session_interrupted_then_resumed_and_completed(db):
    """The load-bearing correctness case: a session that WAS interrupted but was
    then resumed and completed normally has a later non-marker tail, so its
    newest active message is not interrupt_close → it must NOT be nudged."""
    now = time.time()
    db.create_session("sessB", source="cli", model="test-model")
    r1 = db.append_message("sessB", role="assistant", content="was cut", timestamp=now - 100)
    db.update_message_finish_reason("sessB", r1, "interrupt_close")
    # later, resumed and finished normally (newer message, no marker)
    db.append_message("sessB", role="assistant", content="finished normally", timestamp=now)
    assert db.most_recent_interrupt_close_session() is None


def test_returns_none_when_no_interrupted_session(db):
    now = time.time()
    db.create_session("clean", source="cli", model="test-model")
    db.append_message("clean", role="assistant", content="done", timestamp=now)
    assert db.most_recent_interrupt_close_session() is None


def test_recency_bound_excludes_stale_interrupt(db):
    old = time.time() - 48 * 3600  # 2 days ago
    _seed_interrupted(db, "ancient", ts=old)
    assert db.most_recent_interrupt_close_session(within_seconds=24 * 3600) is None
    # without the bound it still surfaces
    assert db.most_recent_interrupt_close_session()["id"] == "ancient"


def test_picks_the_newest_when_several_are_interrupted(db):
    now = time.time()
    _seed_interrupted(db, "older", ts=now - 500)
    _seed_interrupted(db, "newer", ts=now - 10)
    hit = db.most_recent_interrupt_close_session()
    assert hit["id"] == "newer"


def test_tied_timestamp_pick_is_deterministic(db):
    """Two interrupted sessions whose last messages share an identical timestamp
    must resolve to the later-inserted row (higher message id), stably.

    Behavioral check on the returned row PLUS a source-contract assertion that
    the query carries the explicit ``m.id DESC`` tiebreaker — SQLite's ordering
    with only ``m.timestamp DESC`` is *unspecified* on a tie (it may coincidentally
    return the right row on one build/plan and the wrong one on another), so the
    behavioral assert alone can't gate the invariant. The source-contract assert
    fails if the tiebreaker is ever removed.
    """
    import inspect

    src = inspect.getsource(db.most_recent_interrupt_close_session)
    assert "ORDER BY m.timestamp DESC, m.id DESC" in src, (
        "the outer ORDER BY must carry the m.id DESC tiebreaker for a stable "
        "pick on tied timestamps"
    )

    ts = time.time()
    _seed_interrupted(db, "tie-first", ts=ts)
    _seed_interrupted(db, "tie-second", ts=ts)  # same ts, inserted later → higher id
    picks = {db.most_recent_interrupt_close_session()["id"] for _ in range(5)}
    assert picks == {"tie-second"}, picks


def test_source_filter_scopes_to_cli(db):
    now = time.time()
    # a desktop session interrupted — must NOT surface for the cli nudge
    db.create_session("dt", source="desktop", model="test-model")
    r = db.append_message("dt", role="assistant", content="cut", timestamp=now)
    db.update_message_finish_reason("dt", r, "interrupt_close")
    assert db.most_recent_interrupt_close_session(source="cli") is None
    assert db.most_recent_interrupt_close_session(source="desktop")["id"] == "dt"


def test_archived_session_is_excluded(db):
    now = time.time()
    _seed_interrupted(db, "arch", ts=now)
    db._execute_write(
        lambda conn: conn.execute("UPDATE sessions SET archived = 1 WHERE id = ?", ("arch",))
    )
    assert db.most_recent_interrupt_close_session() is None
