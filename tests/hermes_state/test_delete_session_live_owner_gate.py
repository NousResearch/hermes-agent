"""Live-owner gate on ``delete_session`` — the sibling call paths of #125157.

#125157 taught the tui_gateway ``session.delete`` RPC to refuse deleting a session that
is live *in this process* (the registry check), with the FK-trip failure mode spelled out
in its comment. The lease/lock rows it is really about are persisted in the DB and hence
visible **cross-process**, and prune/sweep already refuse under them via
``_write_guards_reject`` — but every other caller reaches the same store method
(``hermes sessions delete``, the dashboard ``DELETE /api/sessions/{id}``, the gateway)
with no guard at all. The contract here: ``delete_session`` itself refuses while a live
turn lease or compression lock holds the session or any delegate child it cascades into,
expired/dead-holder guards are reclaimed as usual, and unguarded sessions delete exactly
as before.
"""

import json
import os
import time

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionInUseError


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _exists(db, session_id):
    with db._read_ctx() as conn:
        return conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is not None


LIVE_HOLDER = f"pid={os.getpid()}"  # same-process holder is never provably dead (#75316 rule)


class TestLiveTurnLease:
    def test_live_lease_refuses_deletion_and_keeps_the_row(self, db):
        db.create_session("s1", source="cli")
        assert db.try_acquire_session_turn_lease("s1", LIVE_HOLDER, ttl_seconds=300)
        with pytest.raises(SessionInUseError):
            db.delete_session("s1")
        assert _exists(db, "s1")
        with db._read_ctx() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE session_id = 's1'").fetchone()["n"]
        assert n == 0  # nothing was written before the refusal

    def test_expired_lease_is_reclaimed_and_deletion_proceeds(self, db):
        db.create_session("s1", source="cli")
        assert db.try_acquire_session_turn_lease("s1", LIVE_HOLDER, ttl_seconds=0.05)
        time.sleep(0.12)
        assert db.delete_session("s1") is True
        assert not _exists(db, "s1")

    def test_dead_holder_lease_is_reclaimed_and_deletion_proceeds(self, db):
        db.create_session("s1", source="cli")
        # A holder PID that provably does not exist: reclaim-on-kernel-proof applies
        # even while expires_at is still in the future.
        assert db.try_acquire_session_turn_lease("s1", "pid=999999999", ttl_seconds=300)
        assert db.delete_session("s1") is True
        assert not _exists(db, "s1")


class TestLiveCompressionLock:
    def test_live_compression_lock_refuses_deletion(self, db):
        db.create_session("s1", source="cli")
        assert db.try_acquire_compression_lock("s1", LIVE_HOLDER, ttl_seconds=300)
        with pytest.raises(SessionInUseError):
            db.delete_session("s1")
        assert _exists(db, "s1")


class TestDelegateChildren:
    def _make_delegate_child(self, db, parent, child):
        db.create_session(child, source="subagent",
                          model_config={"_delegate_from": parent})

    def test_child_live_lease_refuses_parent_deletion(self, db):
        db.create_session("parent", source="cli")
        self._make_delegate_child(db, "parent", "child")
        assert db.try_acquire_session_turn_lease("child", LIVE_HOLDER, ttl_seconds=300)
        with pytest.raises(SessionInUseError):
            db.delete_session("parent")
        assert _exists(db, "parent")
        assert _exists(db, "child")  # cascade never started

    def test_unguarded_child_deletes_with_parent_as_before(self, db):
        db.create_session("parent", source="cli")
        self._make_delegate_child(db, "parent", "child")
        assert db.delete_session("parent") is True
        assert not _exists(db, "parent")
        assert not _exists(db, "child")


class TestUnguardedPaths:
    def test_unguarded_session_deletes_unchanged(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", "user", {"role": "user", "content": "hi"})
        assert db.delete_session("s1") is True
        assert not _exists(db, "s1")

    def test_expected_ids_fence_still_enforced(self, db):
        db.create_session("s1", source="cli")
        # A drifted delegate set must still fail the fence, not delete.
        assert db.delete_session("s1", expected_delete_ids=["s1", "ghost"]) is False
        assert _exists(db, "s1")

    def test_missing_session_still_returns_false(self, db):
        assert db.delete_session("nope") is False
