"""One durable succession record per room: a monotonic fence and one promise per epoch."""

import multiprocessing
import sqlite3
import threading
from contextlib import closing

import pytest

from gateway import hosted_room_fence as fence
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

ROOM = "room-one"


@pytest.fixture
def path(tmp_path):
    value = tmp_path / "runs.db"
    with closing(RunIdempotencyStore(str(value))):
        pass
    return value


def promise(path, candidate, epoch=2, *, room_id=ROOM, now=None):
    return fence.fence_and_promise(path, room_id=room_id, fence_epoch=epoch - 1, promise_epoch=epoch,
                                   candidate_install_id=candidate, now=now)


def test_unfenced_room_has_no_record(path):
    assert fence.room_fence_state(path, ROOM) == {"fenced_epoch": 0, "promise": None, "authority": None}
    assert fence.successor_may_control(path, ROOM, "install:a") is False


def test_promise_is_idempotent_for_its_candidate_and_refused_to_any_other(path):
    first = promise(path, "install:a", now=100.0)
    assert first == {"fenced_epoch": 1, "idempotent": False, "authority": None, "promise": {
        "epoch": 2, "candidate_install_id": "install:a", "issued_at": 100.0}}
    again = promise(path, "install:a", now=200.0)
    assert again == {**first, "idempotent": True}
    with pytest.raises(fence.RoomAuthorityPromised) as refused:
        promise(path, "install:b")
    assert (refused.value.code, refused.value.status) == ("room_authority_promised", 409)
    assert fence.room_fence_state(path, ROOM) == {"fenced_epoch": 1, "promise": first["promise"], "authority": None}
    assert fence.successor_may_control(path, ROOM, "install:a") is True
    assert fence.successor_may_control(path, ROOM, "install:b") is False
    # Another room keeps its own record.
    assert promise(path, "install:b", room_id="room-two")["promise"]["candidate_install_id"] == "install:b"


def test_a_later_epoch_fences_the_earlier_promise_and_never_goes_back(path):
    promise(path, "install:a", 2)
    later = promise(path, "install:b", 3)
    assert later["fenced_epoch"] == 2 and later["promise"]["candidate_install_id"] == "install:b"
    assert fence.successor_may_control(path, ROOM, "install:a") is False
    assert fence.successor_may_control(path, ROOM, "install:b") is True
    for epoch, candidate in ((2, "install:a"), (2, "install:c"), (3, "install:c")):
        with pytest.raises((fence.RoomAuthorityFenced, fence.RoomAuthorityPromised)):
            promise(path, candidate, epoch)
    with pytest.raises(fence.RoomAuthorityFenced) as refused:
        promise(path, "install:a", 2)
    assert (refused.value.code, refused.value.status) == ("room_authority_fenced", 409)
    assert promise(path, "install:b", 3)["idempotent"] is True
    assert fence.room_fence_state(path, ROOM)["fenced_epoch"] == 2


@pytest.mark.parametrize("arguments", [
    {"fence_epoch": 1, "promise_epoch": 3}, {"fence_epoch": 0, "promise_epoch": 1},
    {"fence_epoch": True, "promise_epoch": 2}, {"fence_epoch": "1", "promise_epoch": 2},
    {"fence_epoch": 1, "promise_epoch": 2.0}, {"room_id": " room-one"}, {"room_id": "room\0one"},
    {"room_id": ""}, {"candidate_install_id": "install:a "}, {"candidate_install_id": 7},
])
def test_requests_are_exact(path, arguments):
    request = {"room_id": ROOM, "fence_epoch": 1, "promise_epoch": 2, "candidate_install_id": "install:a",
               **arguments}
    with pytest.raises(ValueError):
        fence.fence_and_promise(path, **request)
    assert fence.room_fence_state(path, ROOM) == {"fenced_epoch": 0, "promise": None, "authority": None}


def test_sql_guards_keep_the_record_monotonic_and_permanent(path):
    promise(path, "install:a", 3)
    statements = (
        f"UPDATE {fence.FENCES} SET fenced_epoch=1",
        f"UPDATE {fence.FENCES} SET candidate_install_id='install:b'",
        f"UPDATE {fence.FENCES} SET issued_at=issued_at+1",
        f"UPDATE {fence.FENCES} SET promise_epoch=2",
        f"UPDATE {fence.FENCES} SET promise_epoch=NULL, candidate_install_id=NULL, issued_at=NULL",
        f"UPDATE {fence.FENCES} SET room_id='room-two'",
        f"DELETE FROM {fence.FENCES}",
    )
    fence.learn_authority(path, room_id=ROOM, epoch=4, install_id="install:a")
    statements += (
        f"UPDATE {fence.FENCES} SET authority_epoch=3",
        f"UPDATE {fence.FENCES} SET authority_install_id='install:b'",
        f"UPDATE {fence.FENCES} SET authority_epoch=NULL, authority_install_id=NULL",
    )
    for statement in statements:
        with closing(sqlite3.connect(path)) as conn, pytest.raises(sqlite3.IntegrityError):
            conn.execute(statement)
    assert fence.room_fence_state(path, ROOM)["promise"]["candidate_install_id"] == "install:a"


def test_capacity_refuses_new_rooms_without_evicting_a_fence(path, monkeypatch):
    monkeypatch.setattr(fence, "MAX_FENCED_ROOMS", 1)
    promise(path, "install:a")
    with pytest.raises(fence.RoomFenceCapacity) as full:
        promise(path, "install:a", room_id="room-two")
    assert full.value.status == 507
    assert promise(path, "install:b", 3)["promise"]["epoch"] == 3
    assert fence.room_fence_state(path, "room-two") == {"fenced_epoch": 0, "promise": None, "authority": None}


def test_unavailable_storage_is_typed(tmp_path):
    with pytest.raises(fence.RoomFenceError) as failure:
        promise(tmp_path, "install:a")
    assert (failure.value.code, failure.value.status) == ("room_fence_unavailable", 503)


def test_concurrent_candidates_get_exactly_one_promise_per_epoch(path):
    barrier = threading.Barrier(8)
    results = {}

    def ask(candidate):
        barrier.wait()
        try:
            results[candidate] = promise(path, candidate)["promise"]["candidate_install_id"]
        except fence.RoomAuthorityPromised as exc:
            results[candidate] = exc.code

    threads = [threading.Thread(target=ask, args=(f"install:{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    winners = [candidate for candidate, outcome in results.items() if outcome == candidate]
    assert len(results) == 8 and len(winners) == 1
    assert sorted(set(results.values()) - {winners[0]}) == ["room_authority_promised"]
    assert fence.room_fence_state(path, ROOM)["promise"]["candidate_install_id"] == winners[0]


def _process_candidate(path, candidate, start, channel):
    start.wait(10)
    try:
        channel.send(promise(path, candidate)["promise"]["candidate_install_id"])
    except fence.RoomFenceError as exc:
        channel.send(exc.code)
    finally:
        channel.close()


def test_separate_processes_get_exactly_one_promise_per_epoch(path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = []
    for number in range(4):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_process_candidate, args=(str(path), f"install:{number}", start, child))
        process.start()
        child.close()
        workers.append((process, parent))
    start.set()
    outcomes = []
    try:
        for _, channel in workers:
            assert channel.poll(15), "worker did not report"
            outcomes.append(channel.recv())
    finally:
        for process, channel in workers:
            process.join(5)
            if process.is_alive():
                process.terminate()
            channel.close()
    winners = [outcome for outcome in outcomes if outcome.startswith("install:")]
    assert len(winners) == 1 and outcomes.count("room_authority_promised") == 3
    assert fence.room_fence_state(path, ROOM)["promise"]["candidate_install_id"] == winners[0]


def test_the_record_survives_restart_of_the_runs_store(path):
    promise(path, "install:a", 4)
    with closing(RunIdempotencyStore(str(path))):
        pass
    assert fence.room_fence_state(path, ROOM)["fenced_epoch"] == 3
    assert promise(path, "install:a", 4)["idempotent"] is True
    with pytest.raises(fence.RoomAuthorityPromised):
        promise(path, "install:b", 4)


def test_a_learned_authority_fences_earlier_epochs_and_takes_control_from_a_losing_promise(path):
    promise(path, "install:loser", 2)
    assert fence.successor_may_control(path, ROOM, "install:loser") is True
    learned = fence.learn_authority(path, room_id=ROOM, epoch=2, install_id="install:winner")
    assert learned["authority"] == {"epoch": 2, "install_id": "install:winner"} and learned["fenced_epoch"] == 1
    assert fence.successor_may_control(path, ROOM, "install:winner") is True
    assert fence.successor_may_control(path, ROOM, "install:loser") is False
    with pytest.raises(fence.RoomAuthorityConflict) as conflict:
        fence.learn_authority(path, room_id=ROOM, epoch=2, install_id="install:loser")
    assert (conflict.value.code, conflict.value.status) == ("room_authority_conflict", 409)
    # An older authority changes nothing; a later promise moves control on again.
    assert fence.learn_authority(path, room_id="room-one", epoch=2, install_id="install:winner") == learned
    promise(path, "install:next", 4)
    assert fence.successor_may_control(path, ROOM, "install:next") is True
    assert fence.successor_may_control(path, ROOM, "install:winner") is False
    later = fence.learn_authority(path, room_id=ROOM, epoch=5, install_id="install:after")
    assert later["fenced_epoch"] == 4 and fence.successor_may_control(path, ROOM, "install:after") is True


def test_an_old_authority_can_fence_its_own_epoch_without_promising(path):
    fenced = fence.fence_room(path, room_id=ROOM, fence_epoch=3)
    assert fenced == {"fenced_epoch": 3, "promise": None, "authority": None}
    assert fence.fence_room(path, room_id=ROOM, fence_epoch=2) == fenced
    with pytest.raises(fence.RoomAuthorityFenced):
        promise(path, "install:a", 3)
    assert promise(path, "install:a", 4)["fenced_epoch"] == 3
    assert fence.successor_may_control(path, ROOM, "install:a") is True
