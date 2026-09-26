"""``SessionDB.session_turn_lease_holder`` — the read side of the turn-lease guard (#123583).

A delete surface must refuse exactly while a lease is *live*, using the same rule the acquire path
uses to reclaim one: not expired AND the holder process not provably gone. Anything stricter would
let a crashed holder block a delete that could equally have reclaimed the lease; anything looser
would delete a row out from under a turn that is mid-write.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from hermes_state import SessionDB


def _holder(tag: str = "turn") -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _dead_pid() -> int:
    """A PID that provably exited — the liveness probe is then kernel proof, not a guess."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _expire_lease(db, session_id: str) -> None:
    db._conn.execute(
        "UPDATE session_turn_leases SET expires_at = ? WHERE conversation_id = ?",
        (time.time() - 5, session_id),
    )
    db._conn.commit()


def _set_end_reason(db, session_id: str, end_reason: str) -> None:
    db._conn.execute("UPDATE sessions SET end_reason = ? WHERE id = ?", (end_reason, session_id))
    db._conn.commit()


def test_no_lease_row_reports_none(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    assert db.session_turn_lease_holder("s1") is None


def test_blank_session_id_reports_none(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    assert db.session_turn_lease_holder("") is None


def test_live_lease_reports_its_holder(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = _holder()
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=300)
    assert db.session_turn_lease_holder("s1") == holder


def test_released_lease_reports_none(tmp_path):
    """The guard must lift as soon as the turn releases — otherwise every later delete is refused."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = _holder()
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=300)
    db.release_session_turn_lease("s1", holder)
    assert db.session_turn_lease_holder("s1") is None


def test_expired_lease_reports_none(tmp_path):
    """An expired row is reclaimable, so it must not block a delete either."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    assert db.try_acquire_session_turn_lease("s1", _holder(), ttl_seconds=300)
    _expire_lease(db, "s1")
    assert db.session_turn_lease_holder("s1") is None


def test_dead_holder_reports_none(tmp_path):
    """A crashed holder's unexpired row must not wedge deletes for the rest of its TTL."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    dead = f"pid={_dead_pid()}:turn=crashed:platform=test"
    assert db.try_acquire_session_turn_lease("s1", dead, ttl_seconds=300)
    assert db.session_turn_lease_holder("s1") is None


def test_compression_continuation_reports_the_lineage_lease(tmp_path):
    """Leases are keyed by the conversation root, so a compression child resolves to it too.

    A delete of the continuation must be refused while the lineage root's turn is running —
    resolving the raw id alone would miss exactly that row.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.create_session("child", source="test", parent_session_id="root")
    _set_end_reason(db, "root", "compression")
    holder = _holder("root-turn")
    assert db.try_acquire_session_turn_lease("root", holder, ttl_seconds=300)
    assert db.session_turn_lease_holder("child") == holder


def test_child_of_a_plain_parent_keeps_its_own_key(tmp_path):
    """The walk only crosses compression continuations — a plain parent/child link is not one,
    so a child must not inherit (and thus refuse on) its parent's lease."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.create_session("child", source="test", parent_session_id="root")
    assert db.try_acquire_session_turn_lease("root", _holder(), ttl_seconds=300)
    assert db.session_turn_lease_holder("child") is None
