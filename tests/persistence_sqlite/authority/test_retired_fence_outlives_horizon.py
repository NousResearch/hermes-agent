"""The retired-session marker is the deletion fence: the prune pass reclaims a deleted session's
tombstones after the retry horizon, but never a user delete's marker, so the id stays deleted.
A housekeeping sweep's marker ages out at the horizon (it retires thousands of ids a pass)."""
import json
import time
from types import SimpleNamespace

import pytest

from gateway.session_results import retain_result
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX, create_conflict
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.housekeeping._retirement import prune_retired_tombstones


def _delete_8_days_ago(holder, epoch, sid):
    snap = get_session(holder, sid)
    at = time.time() - 8 * 86400
    with pytest.MonkeyPatch.context() as clock:
        clock.setattr(time, "time", lambda: at)
        mutate_runtime_session(holder, epoch=epoch, principal_id="human", session_id=sid, request_id="del",
                               operation="delete", payload={}, expected_revision=snap["runtime_revision"],
                               expected_generation=snap["runtime_generation"])


def _create(store, sid, source):
    store.sessions.create(NewSession(session_id=sid, source=source))


def test_a_deleted_session_is_never_recreated_after_the_horizon(store, tmp_path):
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id="owner")
    _create(store, "s", "api_server")
    args = dict(principal_id="api", session_id="s", request_id="r1", payload={"text": "charge card"})
    admit_session_input(holder, epoch=epoch, **args)
    row = claim_session_input(holder, epoch=epoch, session_id="s")
    retain_result(store, epoch=epoch, row=row, result={"result": {"final_response": "done"}, "usage": {}})
    _delete_8_days_ago(holder, epoch, "s")
    store.maintenance.prune_sessions(older_than_days=90, media_root=str(tmp_path))
    assert store.state_meta.get(RETIRED_PREFIX + "s") is not None
    with pytest.raises(AuthorityRefusal, match="not_found"):
        create_session(holder, "s", "unknown")  # a late backfill
    with pytest.raises(AuthorityRefusal, match="not_found"):
        admit_session_input(holder, epoch=epoch, **args)  # the old exact retry
    assert get_session(holder, "s") is None


def test_a_deleted_bot_chat_generation_is_never_reissued(store, tmp_path):
    from gateway.session_bot import _bot_chat_creation_id
    from persistence.authority.bindings import local_session_id
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id="owner")
    authority = SimpleNamespace(profile_id="owned", persistence=store)
    actor = SimpleNamespace(subject="owner")
    sid = local_session_id("owned", "owner", "bot-chat:owned")
    _create(store, sid, "cli")
    _delete_8_days_ago(holder, epoch, sid)
    store.maintenance.prune_sessions(older_than_days=90, media_root=str(tmp_path))
    assert _bot_chat_creation_id(authority, actor) == "bot-chat:owned:1"


def test_a_swept_session_fence_ages_out_at_the_horizon(store, tmp_path):
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id="owner")
    _create(store, "swept", "cron")
    store.sessions.end("swept", "test")
    _create(store, "empty", "cli")
    _create(store, "deleted", "cli")
    _delete_8_days_ago(holder, epoch, "deleted")
    assert store.maintenance.prune_sessions(older_than_days=None, media_root=str(tmp_path)).count == 1
    assert store.sessions.delete_if_empty("empty", media_root=str(tmp_path))  # the /clear and exit cleanup of an unused session
    for sid in ("swept", "empty"):
        assert store.state_meta.get(RETIRED_PREFIX + sid) is not None  # fenced within the horizon
    holder._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
    for sid in ("swept", "empty"):
        assert store.state_meta.get(RETIRED_PREFIX + sid) is None, f"{sid}: a sweep's marker never ages out"
    assert store.state_meta.get(RETIRED_PREFIX + "deleted") is not None


@pytest.mark.parametrize("sid, ages_out", [("user-42", True), ("api-0123456789abcdef", False)],
                         ids=["caller-named", "door-derived"])
def test_a_swept_api_session_keeps_its_fence_only_when_a_door_derived_its_id(store, tmp_path, sid, ages_out):
    holder = store.holder()
    _create(store, sid, "api_server")
    store.sessions.end(sid, "test")
    assert store.maintenance.prune_sessions(older_than_days=None, media_root=str(tmp_path)).count == 1
    holder._execute_write(lambda conn: prune_retired_tombstones(conn, older_than=0))
    assert (store.state_meta.get(RETIRED_PREFIX + sid) is None) is ages_out
    with holder._read_ctx() as conn:
        assert (create_conflict(conn, sid) is None) is ages_out


def test_a_permanent_marker_an_earlier_release_wrote_for_a_swept_api_session_stays(store):
    """An earlier release gave a swept caller-named API id the same permanent marker as a user
    delete (``{"retired_at"}`` only). The two can't be told apart, so the upgraded prune pass keeps
    both: a deleted id must never come back."""
    holder = store.holder()
    store.state_meta.set(RETIRED_PREFIX + "user-42", json.dumps({"retired_at": time.time() - 30 * 86400}))
    holder._execute_write(lambda conn: prune_retired_tombstones(conn))
    assert store.state_meta.get(RETIRED_PREFIX + "user-42") is not None
    with holder._read_ctx() as conn:
        assert create_conflict(conn, "user-42") == ("session_deleted", "Session was deleted; its id cannot be reused")
