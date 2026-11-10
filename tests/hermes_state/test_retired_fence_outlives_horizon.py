"""The retired-session marker is the deletion fence: the prune pass reclaims a deleted session's
tombstones after the retry horizon, but never a user delete's marker, so the id stays deleted.
A housekeeping sweep's marker ages out at the horizon (it retires thousands of ids a pass)."""
import time
from types import SimpleNamespace

import pytest

import hermes_state_runtime as rt
from gateway.session_results import retain_result
from hermes_state import SessionDB


def _delete_8_days_ago(db, epoch, sid):
    snap = db.get_session(sid)
    at = time.time() - 8 * 86400
    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(time, "time", lambda: at)
        rt.mutate_runtime_session(db, epoch=epoch, principal_id="human", session_id=sid, request_id="del",
                                  operation="delete", payload={}, expected_revision=snap["runtime_revision"],
                                  expected_generation=snap["runtime_generation"])


def test_a_deleted_session_is_never_recreated_after_the_horizon(tmp_path):
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    with SessionDB(tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        db.create_session("s", source="api_server")
        args = dict(principal_id="api", session_id="s", request_id="r1", payload={"text": "charge card"})
        rt.admit_session_input(db, epoch=epoch, **args)
        row = rt.claim_session_input(db, epoch=epoch, session_id="s")
        retain_result(db, epoch=epoch, row=row, result={"result": {"final_response": "done"}, "usage": {}})
        _delete_8_days_ago(db, epoch, "s")
        db.prune_sessions(older_than_days=90)
        assert db.get_meta(RETIRED_PREFIX + "s") is not None
        with pytest.raises(rt.RuntimeStoreError, match="not_found"):
            db.ensure_session("s", source="unknown")  # a late backfill
        with pytest.raises(rt.RuntimeStoreError, match="not_found"):
            rt.admit_session_input(db, epoch=epoch, **args)  # the old exact retry
        assert db.get_session("s") is None


def test_a_deleted_bot_chat_generation_is_never_reissued(tmp_path):
    from gateway.session_bot import _bot_chat_creation_id
    from gateway.session_local_recovery import local_identity
    with SessionDB(tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        authority, actor = SimpleNamespace(profile_id="owned", db=db), SimpleNamespace(subject="owner")
        sid = local_identity("owned", "owner", "bot-chat:owned")
        db.create_session(sid, source="cli")
        _delete_8_days_ago(db, epoch, sid)
        db.prune_sessions(older_than_days=90)
        assert _bot_chat_creation_id(authority, actor) == "bot-chat:owned:1"


def test_a_swept_session_fence_ages_out_at_the_horizon(tmp_path):
    from hermes_state_mutation_retirement import RETIRED_PREFIX, prune_retired_tombstones
    with SessionDB(tmp_path / "state.db") as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        db.create_session("swept", source="cron")
        db.end_session("swept", "test")
        db.create_session("empty", source="cli")
        db.create_session("deleted", source="cli")
        _delete_8_days_ago(db, epoch, "deleted")
        assert db.prune_sessions(older_than_days=None) == 1
        assert db.delete_session_if_empty("empty")  # the /clear and exit cleanup of an unused session
        for sid in ("swept", "empty"):
            assert db.get_meta(RETIRED_PREFIX + sid) is not None  # fenced within the horizon
        db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        for sid in ("swept", "empty"):
            assert db.get_meta(RETIRED_PREFIX + sid) is None, f"{sid}: a sweep's marker never ages out"
        assert db.get_meta(RETIRED_PREFIX + "deleted") is not None


@pytest.mark.parametrize("sid, ages_out", [("user-42", True), ("api-0123456789abcdef", False)],
                         ids=["caller-named", "door-derived"])
def test_a_swept_api_session_keeps_its_fence_only_when_a_door_derived_its_id(tmp_path, sid, ages_out):
    from hermes_state_mutation_retirement import RETIRED_PREFIX, create_conflict, prune_retired_tombstones
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(sid, source="api_server")
        db.end_session(sid, "test")
        assert db.prune_sessions(older_than_days=None) == 1
        db._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
        assert (db.get_meta(RETIRED_PREFIX + sid) is None) is ages_out
        with db._read_ctx() as conn:
            assert (create_conflict(conn, sid) is None) is ages_out


def test_a_permanent_marker_an_earlier_release_wrote_for_a_swept_api_session_stays(tmp_path):
    """An earlier release gave a swept caller-named API id the same permanent marker as a user
    delete (``{"retired_at"}`` only). The two can't be told apart, so the upgraded prune pass keeps
    both: a deleted id must never come back."""
    import json
    from hermes_state_mutation_retirement import RETIRED_PREFIX, create_conflict, prune_retired_tombstones
    with SessionDB(tmp_path / "state.db") as db:
        db.set_meta(RETIRED_PREFIX + "user-42", json.dumps({"retired_at": time.time() - 30 * 86400}))
        db._execute_write(lambda conn: prune_retired_tombstones(conn))
        assert db.get_meta(RETIRED_PREFIX + "user-42") is not None
        with db._read_ctx() as conn:
            assert create_conflict(conn, "user-42") == ("session_deleted", "Session was deleted; its id cannot be reused")
