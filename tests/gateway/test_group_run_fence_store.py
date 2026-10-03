"""The succession fence at the Runs store's accepting writer, beside the owner freeze."""

import json
import multiprocessing
import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_room_fence as fence
from gateway.platforms import api_server_run_idempotency as storage
from gateway.platforms.api_server_run_scope import room_run_scope_key
from tests.gateway.test_group_run_scope import IDENTITY

SUCCESSOR = {**IDENTITY, "home_install_id": "install:successor", "authority_gateway_id": "install:successor",
             "authority_epoch": IDENTITY["authority_epoch"] + 1}


@pytest.fixture
def store(tmp_path):
    value = storage.RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        yield value
    finally:
        value.close()


def reserve(store, run_id="run-one", *, identity=IDENTITY, status="queued", key=None):
    return store.reserve(room_run_scope_key(identity), key or "key-" + run_id, "fingerprint-" + run_id, run_id,
                         {"status": status}, owner_pid=123, owner_started=456, identity=identity)


def promise(path, candidate=SUCCESSOR["authority_gateway_id"], epoch=SUCCESSOR["authority_epoch"]):
    return fence.fence_and_promise(path, room_id=IDENTITY["room_id"], fence_epoch=epoch - 1,
                                   promise_epoch=epoch, candidate_install_id=candidate)


def test_fenced_epochs_refuse_new_admission_but_replay_existing_runs(store):
    assert reserve(store)[0] == "created"
    promise(store.path)
    assert reserve(store)[0] == "reused"
    with pytest.raises(fence.RoomAuthorityFenced) as refused:
        reserve(store, "new-run")
    assert (refused.value.code, refused.value.status) == ("room_authority_fenced", 409)
    older = {**IDENTITY, "authority_epoch": 1, "member_id": "never-admitted"}
    with pytest.raises(fence.RoomAuthorityFenced):
        reserve(store, "older-run", identity=older)
    # The promised epoch, other rooms and ordinary runs stay open.
    assert reserve(store, "successor-run", identity=SUCCESSOR)[0] == "created"
    assert reserve(store, "other-room", identity={**IDENTITY, "room_id": "room-two"})[0] == "created"
    assert store.reserve("0" * 64, "plain", "fp", "plain-run", {"status": "queued"})[0] == "created"
    status = store.status_for_run(room_run_scope_key(IDENTITY), "run-one")
    assert status["status"] == {"status": "queued"}


def test_existing_runs_keep_their_status_updates_and_listing(store):
    reserve(store)
    promise(store.path)
    store.update_status("run-one", {"status": "running"})
    assert store.status_for_run(room_run_scope_key(IDENTITY), "run-one")["status"] == {"status": "running"}
    listing = store.list_room_scopes(target_install_id=IDENTITY["target_install_id"],
                                     target_profile=IDENTITY["target_profile"])
    assert [item["identity"] for item in listing["participants"]] == [IDENTITY]


def test_the_owner_freeze_is_unaffected_by_a_fence(store):
    reserve(store)
    promise(store.path)
    snapshot = store.freeze_room_scope(IDENTITY, "owner-stop")
    assert snapshot["identity"] == IDENTITY and snapshot["counts"]["nonterminal"] == 1
    # A frozen scope keeps the freeze's own refusal and control decision.
    with pytest.raises(storage.GroupRunFrozen):
        reserve(store, "new-run")
    with store.group_control_open(room_run_scope_key(IDENTITY)) as allowed:
        assert allowed is False
    # A fence elsewhere never freezes, and freezing never fences.
    reserve(store, "successor-run", identity=SUCCESSOR)
    store.freeze_room_scope(SUCCESSOR, "owner-stop-successor")
    assert fence.room_fence_state(store.path, IDENTITY["room_id"])["fenced_epoch"] == IDENTITY["authority_epoch"]
    assert store.room_stop_snapshot("owner-stop")["frozen_at"] == snapshot["frozen_at"]


def test_controls_from_a_fenced_epoch_are_refused_at_the_writer(store):
    reserve(store)
    scope = room_run_scope_key(IDENTITY)
    with store.group_control_open(scope) as allowed:
        assert allowed is True
    promise(store.path)
    for freeze in (True, False):
        with pytest.raises(fence.RoomAuthorityFenced), store.group_control_open(scope, freeze=freeze):
            pytest.fail("a fenced epoch must not reach control")
    reserve(store, "successor-run", identity=SUCCESSOR)
    with store.group_control_open(room_run_scope_key(SUCCESSOR)) as allowed:
        assert allowed is True
    # An unrecorded scope has no room to fence.
    with store.group_control_open("0" * 64) as allowed:
        assert allowed is True


def test_only_the_promised_successor_controls_existing_runs(store):
    reserve(store)
    assert store.successor_run_scope("run-one", successor=SUCCESSOR) is None
    promise(store.path)
    scope = room_run_scope_key(IDENTITY)
    assert store.successor_run_scope("run-one", successor=SUCCESSOR) == scope
    for field, value in (("authority_gateway_id", "install:other"), ("authority_epoch", 5),
                         ("member_id", "other-member"), ("room_id", "room-two"),
                         ("target_install_id", "other-target"), ("target_profile", "other-profile")):
        assert store.successor_run_scope("run-one", successor={**SUCCESSOR, field: value}) is None
    assert store.successor_run_scope("missing-run", successor=SUCCESSOR) is None
    # A run the successor itself admitted is its own, never "existing" work it inherited.
    reserve(store, "successor-run", identity=SUCCESSOR)
    assert store.successor_run_scope("successor-run", successor=SUCCESSOR) is None
    # A later promise moves control on.
    later = {**SUCCESSOR, "authority_gateway_id": "install:later", "authority_epoch": 5}
    promise(store.path, later["authority_gateway_id"], 5)
    assert store.successor_run_scope("run-one", successor=SUCCESSOR) is None
    assert store.successor_run_scope("run-one", successor=later) == scope
    assert store.successor_run_scope("successor-run", successor=later) == room_run_scope_key(SUCCESSOR)


def test_a_learned_certified_authority_takes_control_from_a_losing_promise(store):
    reserve(store)
    loser = {**SUCCESSOR, "authority_gateway_id": "install:loser"}
    promise(store.path, loser["authority_gateway_id"])
    assert store.successor_run_scope("run-one", successor=loser) == room_run_scope_key(IDENTITY)
    fence.learn_authority(store.path, room_id=IDENTITY["room_id"], epoch=SUCCESSOR["authority_epoch"],
                          install_id=SUCCESSOR["authority_gateway_id"])
    assert store.successor_run_scope("run-one", successor=loser) is None
    assert store.successor_run_scope("run-one", successor=SUCCESSOR) == room_run_scope_key(IDENTITY)


def test_unreadable_scope_evidence_never_authorizes_a_successor(store):
    reserve(store)
    promise(store.path)
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("UPDATE group_run_scopes SET identity_json=?",
                     (json.dumps({**IDENTITY, "room_id": "room-two"}),))
    assert store.successor_run_scope("run-one", successor=SUCCESSOR) is None
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("UPDATE group_run_scopes SET identity_json='not json'")
    assert store.successor_run_scope("run-one", successor=SUCCESSOR) is None


def test_sql_guards_back_the_admission_check_for_other_writers(store):
    reserve(store)
    promise(store.path)
    scope = room_run_scope_key(IDENTITY)
    with closing(sqlite3.connect(store.path)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""INSERT INTO run_idempotency(scope,idempotency_key,fingerprint,run_id,status_json,
                created_at,updated_at) VALUES (?,?,?,?,?,1,1)""", (scope, "old-writer", "fp", "old-run", "{}"))
        other = {**IDENTITY, "member_id": "unrecorded"}
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO group_run_scopes VALUES (?,?,1,1)",
                         (room_run_scope_key(other), json.dumps(other)))
        conn.execute("INSERT INTO group_run_scopes VALUES (?,?,1,1)",
                     (room_run_scope_key(SUCCESSOR), json.dumps(SUCCESSOR)))
        conn.commit()


def test_fence_survives_restart_of_the_store(store):
    reserve(store)
    promise(store.path)
    store.close()
    with closing(storage.RunIdempotencyStore(str(store.path))) as restarted:
        with pytest.raises(fence.RoomAuthorityFenced):
            reserve(restarted, "after-restart")
        assert restarted.successor_run_scope("run-one", successor=SUCCESSOR) == room_run_scope_key(IDENTITY)


def _actor(path, operation, pause, channel):
    store = None
    try:
        store = storage.RunIdempotencyStore(path)
        held = False

        def hold_writer(sql):
            nonlocal held
            prefix = "INSERT INTO run_idempotency(" if operation == "reserve" else "INSERT INTO hosted_room_fences"
            if pause and not held and sql.startswith(prefix):
                held = True
                channel.send({"state": "holding_writer"})
                if channel.poll(10):
                    channel.recv()

        store._conn.set_trace_callback(hold_writer)
        connect = fence._connect

        def traced(db_path):
            conn = connect(db_path)
            conn.set_trace_callback(hold_writer)
            return conn

        fence._connect = traced
        channel.send({"state": "ready"})
        if not channel.poll(10):
            raise RuntimeError("start signal timed out")
        channel.recv()
        channel.send({"state": "attempting"})
        if operation == "reserve":
            result = {"outcome": reserve(store, "late-run")[0]}
        else:
            result = {"candidate": promise(path)["promise"]["candidate_install_id"]}
        channel.send({"state": "done", "result": result})
    except fence.RoomFenceError as exc:
        channel.send({"state": "done", "error": exc.code})
    except Exception as exc:
        channel.send({"state": "unexpected", "error": type(exc).__name__})
    finally:
        if store is not None:
            store.close()
        channel.close()


def _receive(channel, state):
    assert channel.poll(15), f"worker did not report {state}"
    result = channel.recv()
    assert result["state"] == state, result
    return result


@pytest.mark.parametrize("first", ["fence", "reserve"])
def test_one_writer_orders_an_old_epoch_admission_and_the_fence(tmp_path, first):
    path = tmp_path / "runs.db"
    with closing(storage.RunIdempotencyStore(str(path))) as seeded:
        reserve(seeded, "seed")
    second = "reserve" if first == "fence" else "fence"
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for operation, pause in ((first, True), (second, False)):
            parent, child = context.Pipe()
            process = context.Process(target=_actor, args=(str(path), operation, pause, child))
            process.start()
            child.close()
            workers.append((process, parent))
            _receive(parent, "ready")
        channels = [channel for _, channel in workers]
        channels[0].send("go")
        _receive(channels[0], "attempting")
        _receive(channels[0], "holding_writer")
        channels[1].send("go")
        _receive(channels[1], "attempting")
        channels[0].send("release")
        results = dict(zip((first, second), (_receive(channel, "done") for channel in channels)))
    finally:
        for process, channel in workers:
            try:
                channel.send("release")
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            channel.close()
    assert results["fence"]["result"] == {"candidate": SUCCESSOR["authority_gateway_id"]}
    admitted = first == "reserve"
    assert results["reserve"] == ({"state": "done", "result": {"outcome": "created"}} if admitted
                                  else {"state": "done", "error": "room_authority_fenced"})
    with closing(storage.RunIdempotencyStore(str(path))) as restarted:
        assert restarted.owns_run(room_run_scope_key(IDENTITY), "late-run") is admitted
        assert restarted.successor_run_scope("seed", successor=SUCCESSOR) == room_run_scope_key(IDENTITY)
        with pytest.raises(fence.RoomAuthorityFenced):
            reserve(restarted, "after")
