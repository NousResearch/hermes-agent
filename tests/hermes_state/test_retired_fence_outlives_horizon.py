"""The retired-session marker is the deletion fence: the prune pass reclaims a deleted session's
tombstones after the retry horizon, but never the marker, so the id stays deleted."""
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
