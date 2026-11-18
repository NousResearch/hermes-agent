"""A claimed turn's write to the SQLite ``'shared'`` hosted-room placement is held to the turn's fence like any other
write of the store that issued the claim: a superseded turn publishes nothing there, a live one writes. Outside a
claimed turn the shared write takes no lock of the profile store."""

from __future__ import annotations

import contextvars
import threading

import pytest

import persistence_assembly as pa
from agent.runtime_scope import bind_turn, claim_fence
from persistence import SessionId, TurnFenceSuperseded
from persistence.conversation.sessions import NewSession
from tests.hosted_persistence.test_local_store_execution_paths import _create, _shared_ids


def _claimed(handle, session):
    handle.sessions.create(NewSession(session_id=session, source="cli"))
    # The shared schema is ready before the claim: the fenced target is an ordinary row write.
    _create(handle.hosted_rooms, "existing-room")
    epoch = handle.runtime.begin_epoch(instance_id="owner-1")
    handle.admissions.admit(epoch=epoch, principal_id="p", session_id=session, request_id="r1", payload={})
    return handle.admissions.claim(epoch=epoch, session_id=session)


def test_a_superseded_turns_shared_placement_write_publishes_nothing(tmp_path):
    handle = pa.open_persistence(pa.profile_scope(home=tmp_path), eager=True)
    try:
        session = SessionId("claimed")
        claim = _claimed(handle, session)
        contextvars.Context().run(lambda: handle.runtime.begin_epoch(instance_id="owner-2"))
        with bind_turn(session, claim.admission_id, claim_fence(claim), store=handle.scope):
            with pytest.raises(TurnFenceSuperseded):
                handle.sessions.update_cwd(session, "/stale")
            with pytest.raises(TurnFenceSuperseded):
                _create(handle.hosted_rooms, "stale-created-room")
        assert _shared_ids(handle.hosted_rooms) == ["existing-room"]
    finally:
        handle.close()


def test_a_live_turns_shared_placement_write_lands(tmp_path):
    handle = pa.open_persistence(pa.profile_scope(home=tmp_path), eager=True)
    try:
        session = SessionId("claimed")
        claim = _claimed(handle, session)
        with bind_turn(session, claim.admission_id, claim_fence(claim), store=handle.scope):
            _create(handle.hosted_rooms, "live-room")
            assert sorted(_shared_ids(handle.hosted_rooms)) == ["existing-room", "live-room"]
        assert sorted(_shared_ids(handle.hosted_rooms)) == ["existing-room", "live-room"]
    finally:
        handle.close()


def test_a_shared_placement_write_outside_a_claimed_turn_takes_no_profile_store_lock(tmp_path):
    """Outside a claimed turn the shared write runs as it always has: it never waits on the profile store's writer."""
    handle = pa.open_persistence(pa.profile_scope(home=tmp_path), eager=True)
    try:
        _create(handle.hosted_rooms, "existing-room")
        holder = handle._bound_store().holder()
        done = threading.Event()
        errors = []

        def write():
            try:
                _create(handle.hosted_rooms, "unlocked-room")
            except BaseException as exc:  # reported below
                errors.append(exc)
            finally:
                done.set()
        with holder._lock:
            worker = threading.Thread(target=write, daemon=True)
            worker.start()
            assert done.wait(5.0), "the shared write waited on the profile store's writer lock"
        worker.join()
        assert errors == []
        assert sorted(_shared_ids(handle.hosted_rooms)) == ["existing-room", "unlocked-room"]
    finally:
        handle.close()
