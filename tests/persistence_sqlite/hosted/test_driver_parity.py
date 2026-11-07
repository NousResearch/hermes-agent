"""The SQLite ``HostedRoomDriverRepository`` against main's own driver functions (gateway/hosted_room_driver.py),
read from the pinned main checkout.

Each chain seeds one placement's file through the room verbs, copies it to a twin, then runs every step twice: main's
function on the twin and the backend verb on the original. After every step the outcomes (records projected to
plain values, errors by class name and message) and the driver and work-record tables of both files must be equal,
so the work-record capture each transition runs is compared row for row too. Each step also names the outcome it
expects, so a chain cannot pass by both sides failing the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from persistence.gateway import hosted_rooms as contract
from persistence.sqlite.gateway.hosted_room_driver import SQLiteHostedRoomDriverRepository
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store
from tests.persistence_sqlite.hosted._task_fixtures import (
    DRIVER_TABLES, GATEWAY, ROOM, WORK_TABLES, Hosted, dump, m1_room, make_hosted, plain, raw, schema, seed_event,
    seed_room, status, twin, work_records,
)

__all__ = ["legacy_store"]

TABLES = (*DRIVER_TABLES, *WORK_TABLES)
READS = {"get_task", "list_tasks", "get_task_for_turn"}


@pytest.fixture
def hosted() -> Hosted:
    hosted = make_hosted()
    yield hosted
    hosted.backend.close()


@pytest.fixture
def m1(legacy_store):
    """Main's ``gateway.hosted_room_driver`` from the pinned checkout, importable for the whole test."""
    return legacy_import("gateway.hosted_room_driver")


@dataclass(frozen=True)
class Ident:
    """A task identity built for each side (main's ``TaskIdentity`` or the contract's ``HostedTaskIdentity``)."""

    n: int
    task: str | None = None

    def build(self, cls: type) -> Any:
        return cls(ROOM, self.task or f"task-{self.n}", f"thread-{self.n}", f"turn-{self.n}")


@dataclass(frozen=True)
class Ref:
    """A value an earlier step saved on the same side (a lease or an attempt)."""

    name: str


@dataclass
class Step:
    verb: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    now: float | None = None
    expect: str = "ok"
    save: str | None = None


@dataclass
class Seed:
    """A change made on both files between steps: ``fn`` on the twin, and ``verb(hosted, placement)`` through the
    storage verbs on the placement's file, or ``fn`` there too when it is raw SQL."""

    fn: Callable[[Path], Any]
    verb: Callable[[Hosted, str], Any] | None = None


def _value(item: Any, cls: type, saved: dict[str, Any]) -> Any:
    if isinstance(item, Ident):
        return item.build(cls)
    if isinstance(item, Ref):
        return saved[item.name]
    return item


def _call(call: Callable[[], Any]) -> tuple[tuple, Any]:
    """The step's outcome (``status``-able, plain) and its raw value."""
    try:
        value = call()
    except Exception as exc:  # compared by class name and message
        return ("error", type(exc).__name__, str(exc)), None
    return ("ok", plain(value)), value


def run_chain(hosted: Hosted, tmp_path: Path, placement: str, steps: list[Step | Seed]) -> list[tuple]:
    """Run ``steps`` on main (twin file) and the backend (placement's file); returns the backend outcomes. The caller
    holds the legacy scope."""
    m1 = legacy_import("gateway.hosted_room_driver")
    path = hosted.prepare(placement)
    seed_room(hosted, placement)
    # Source events for the tasks, so the work-record capture records them instead of an unavailable view.
    for n in range(1, 6):
        seed_event(hosted, placement, event_id=f"e-{n}", kind="message.user",
                   payload={"text": "q", "thread_id": f"thread-{n}"}, now=100.0 + n / 10)
    legacy_path = twin(path, tmp_path / "m1")
    repo = SQLiteHostedRoomDriverRepository(hosted.store)
    saved: dict[str, dict[str, Any]] = {"m1": {}, "backend": {}}
    results = []
    for index, step in enumerate(steps):
        if isinstance(step, Seed):
            step.fn(legacy_path)
            if step.verb is None:
                step.fn(path)
            else:
                step.verb(hosted, placement)
            continue
        sides = {"m1": m1.TaskIdentity, "backend": contract.HostedTaskIdentity}
        args = {side: [_value(item, cls, saved[side]) for item in step.args] for side, cls in sides.items()}
        kwargs = {side: {key: _value(item, cls, saved[side]) for key, item in step.kwargs.items()}
                  for side, cls in sides.items()}
        if step.verb not in READS:
            kwargs["m1"]["clock"] = lambda now=step.now: now
            kwargs["backend"]["now"] = step.now
        legacy, legacy_value = _call(lambda: getattr(m1, step.verb)(legacy_path, *args["m1"], **kwargs["m1"]))
        backend, backend_value = _call(
            lambda: getattr(repo, step.verb)(*args["backend"], placement=placement, **kwargs["backend"]))
        label = f"step {index} {step.verb}"
        assert backend == legacy, label
        assert status(backend) == step.expect, (label, backend)
        assert dump(path, TABLES) == dump(legacy_path, TABLES), label
        if step.save is not None:
            saved["m1"][step.save], saved["backend"][step.save] = legacy_value, backend_value
        results.append(backend)
    return results


def _lease(process: str = "proc-a", *, now: float, ttl: float = 60.0, epoch: int = 1, save: str | None = None,
           expect: str = "ok") -> Step:
    return Step("acquire_lease", kwargs=dict(room_id=ROOM, gateway_id=GATEWAY, authority_epoch=epoch,
                                             process_generation=process, ttl_seconds=ttl),
                now=now, save=save, expect=expect)


def _payload(seq: int, **extra: Any) -> dict:
    return {"target_profile": "p1", "target_member_id": "m1", "prompt": "hello", "source_event_seq": seq, **extra}


def _admit(n: int, *, now: float, expect: str = "ok", task: str | None = None, **payload: Any) -> Step:
    return Step("admit_task", (Ident(n, task),), dict(payload=_payload(n, **payload)), now=now, expect=expect)


def _start(n: int, lease: str, *, now: float, save: str | None = None, expect: str = "ok", cancel: int = 0) -> Step:
    return Step("start_task", (Ident(n), Ref(lease)), dict(expected_cancel_generation=cancel), now=now, save=save,
                expect=expect)


def _settle(attempt: str, *, now: float, result: Any = None, settlement: str = "settle-1", status_: str = "settled",
            expect: str = "ok") -> Step:
    return Step("settle_task", (Ref(attempt),), dict(settlement_id=settlement, status=status_,
                                                    result={"text": "done"} if result is None else result),
                now=now, expect=expect)


def _generations(verb: str, n: int, lease: str, execution: int, cancel: int, *, now: float, expect: str = "ok",
                 **extra: Any) -> Step:
    return Step(verb, (Ident(n), Ref(lease)), dict(expected_execution_generation=execution,
                                                  expected_cancel_generation=cancel, **extra), now=now, expect=expect)


@pytest.mark.parametrize("placement", ["profile", "shared"])
def test_leases_admission_start_settlement_cancellation_and_reads(hosted, tmp_path, placement, legacy_store):
    results = run_chain(hosted, tmp_path, placement, [
        _lease(now=100.0, save="lease"),
        _lease(now=120.0, ttl=10.0),
        _lease("proc-b", now=150.0, expect="LeaseHeldError"),
        _lease(now=150.0, epoch=2, expect="StaleLeaseError"),
        _admit(1, now=101.0),
        _admit(1, now=102.0),
        Step("admit_task", (Ident(1),), dict(payload=_payload(1, prompt="other")), now=103.0,
             expect="TaskConflictError"),
        _admit(1, task="task-9", now=103.0, expect="TaskConflictError"),
        _admit(2, now=104.0),
        _admit(3, now=105.0, target_profile="p2", target_member_id="m2",
               input_context={"watermark": 0, "event_seqs": [1, 2]}),
        _start(2, "lease", now=110.0, expect="InvalidTaskTransitionError"),
        _start(1, "lease", now=110.0, save="attempt1"),
        _start(2, "lease", now=111.0, expect="InvalidTaskTransitionError"),
        Step("release_lease", (Ref("lease"),), now=112.0, expect="InvalidTaskTransitionError"),
        Step("renew_lease", (Ref("lease"),), dict(ttl_seconds=60.0), now=130.0, save="lease"),
        _settle("attempt1", now=131.0),
        _settle("attempt1", now=132.0),
        _settle("attempt1", now=133.0, result={"text": "other"}, expect="TaskConflictError"),
        Step("get_task", (Ident(1),)),
        Step("get_task", (Ident(1, "task-9"),), expect="TaskConflictError"),
        Step("list_tasks", kwargs=dict(room_id=ROOM)),
        Step("list_tasks", kwargs=dict(room_id=ROOM, status="queued")),
        Step("get_task_for_turn", (Ident(2, "other"),)),
        Step("get_task_for_turn", (Ident(7),)),
        Step("cancel_task", (Ident(2),), dict(cancel_id="cancel-2", expected_cancel_generation=0), now=140.0),
        Step("cancel_task", (Ident(2),), dict(cancel_id="cancel-2", expected_cancel_generation=0), now=141.0),
        Step("cancel_task", (Ident(1),), dict(cancel_id="cancel-1", expected_cancel_generation=0), now=141.0,
             expect="InvalidTaskTransitionError"),
        _start(3, "lease", now=142.0, save="attempt3"),
        Step("cancel_task", (Ident(3),), dict(cancel_id="cancel-3", expected_cancel_generation=0), now=143.0,
             expect="InvalidTaskTransitionError"),
        Step("requeue_not_admitted_task", (Ref("attempt3"),), now=144.0),
        Step("requeue_not_admitted_task", (Ref("attempt3"),), now=145.0),
        _start(3, "lease", now=146.0, save="attempt3"),
        Step("begin_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=0), now=147.0),
        Step("begin_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=0), now=148.0),
        _settle("attempt3", now=149.0, settlement="settle-3", expect="StaleTaskError"),
        _generations("settle_stopping_task", 3, "lease", 2, 1, now=150.0, settlement_id="settle-3", status="failed",
                     result={"error": "late"}),
        _generations("settle_stopping_task", 3, "lease", 2, 1, now=151.0, settlement_id="settle-3", status="failed",
                     result={"error": "late"}),
        Step("complete_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=1), now=152.0,
             expect="StaleTaskError"),
        Step("prune_published_terminal_tasks", kwargs=dict(room_id=ROOM, retain=0), now=153.0),
        Step("release_lease", (Ref("lease"),), now=154.0),
        Step("release_lease", (Ref("lease"),), now=155.0),
        Step("renew_lease", (Ref("lease"),), dict(ttl_seconds=60.0), now=156.0, expect="StaleLeaseError"),
        _lease("proc-b", now=157.0, save="lease2"),
        _lease(now=158.0, expect="LeaseHeldError"),
    ])
    # The replays report themselves.
    assert results[5][1]["idempotent"] is True and results[16][1]["idempotent"] is True
    assert results[-2][1]["reclaimed"] is True
    # The capture both sides compared recorded the tasks, not an unavailable view.
    (record,) = work_records(hosted.path(placement))
    assert record["availability"] == "available" and [task["phase"] for task in record["tasks"]] == [
        "settled", "cancelled", "failed"]


def test_recovery_deferral_requeue_and_resolution(hosted, tmp_path, legacy_store):
    run_chain(hosted, tmp_path, "profile", [
        _lease(now=100.0, save="lease1"),
        _admit(1, now=101.0),
        _admit(2, now=102.0),
        _start(1, "lease1", now=103.0, save="attempt1"),
        _lease("proc-b", now=200.0, save="lease2"),
        _settle("attempt1", now=201.0, expect="StaleLeaseError"),
        Step("recover_room", (Ref("lease1"),), now=201.0, expect="StaleLeaseError"),
        Step("recover_room", (Ref("lease2"),), now=202.0),
        Step("recover_room", (Ref("lease2"),), now=203.0),
        _start(2, "lease2", now=204.0, expect="InvalidTaskTransitionError"),
        _generations("defer_indeterminate_task", 1, "lease2", 1, 1, now=205.0, reason="member-unavailable",
                     expect="StaleTaskError"),
        _generations("defer_indeterminate_task", 1, "lease2", 1, 0, now=205.0, reason="member-unavailable"),
        _generations("defer_indeterminate_task", 1, "lease2", 1, 0, now=206.0, reason="member-unavailable"),
        _generations("requeue_indeterminate_task", 1, "lease2", 1, 0, now=207.0, expect="StaleTaskError"),
        _generations("requeue_deferred_task", 1, "lease2", 1, 0, now=208.0),
        _start(1, "lease2", now=209.0, save="attempt1"),
        _lease("proc-c", now=300.0, save="lease3"),
        Step("recover_room", (Ref("lease3"),), now=301.0),
        _generations("requeue_indeterminate_task", 1, "lease3", 2, 0, now=302.0),
        _start(1, "lease3", now=303.0, save="attempt1"),
        _lease("proc-d", now=400.0, save="lease4"),
        Step("recover_room", (Ref("lease4"),), now=401.0),
        _generations("resolve_indeterminate_task", 1, "lease4", 3, 0, now=402.0, settlement_id="settle-1",
                     status="settled", result={"text": "reconciled"}),
        _generations("resolve_indeterminate_task", 1, "lease4", 3, 0, now=403.0, settlement_id="settle-1",
                     status="settled", result={"text": "reconciled"}),
        _generations("resolve_indeterminate_task", 1, "lease4", 3, 0, now=404.0, settlement_id="settle-2",
                     status="failed", result={}, expect="TaskConflictError"),
        _start(2, "lease4", now=405.0, save="attempt2"),
        Step("begin_task_cancel", (Ident(2),), dict(cancel_id="stop-2", expected_cancel_generation=1), now=406.0,
             expect="StaleTaskError"),
        _lease("proc-e", now=500.0, save="lease5"),
        Step("recover_room", (Ref("lease5"),), now=501.0),
        _generations("resolve_indeterminate_cancellation", 2, "lease5", 1, 0, now=502.0, cancel_id="cancel-2"),
        _generations("resolve_indeterminate_cancellation", 2, "lease5", 1, 0, now=503.0, cancel_id="cancel-2"),
        _generations("resolve_indeterminate_cancellation", 2, "lease5", 1, 0, now=504.0, cancel_id="cancel-3",
                     expect="StaleTaskError"),
        _admit(3, now=505.0),
        Step("begin_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=0), now=506.0,
             expect="InvalidTaskTransitionError"),
        _start(3, "lease5", now=507.0, save="attempt3"),
        Step("begin_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=0), now=508.0),
        Step("complete_task_cancel", (Ident(3),), dict(cancel_id="stop-x", expected_cancel_generation=1), now=509.0,
             expect="StaleTaskError"),
        Step("complete_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=1), now=510.0),
        Step("complete_task_cancel", (Ident(3),), dict(cancel_id="stop-3", expected_cancel_generation=1), now=511.0),
        Step("list_tasks", kwargs=dict(room_id=ROOM)),
    ])


def _checkpoint() -> Seed:
    """The policy checkpoint's tables, as main's hosted-room service creates them when it constructs its checkpoint
    and the first policy verb creates them on the backend."""
    return Seed(
        lambda path: legacy_import("gateway.hosted_room_policy_checkpoint").HostedRoomPolicyCheckpoint(path),
        lambda hosted, placement: hosted.store.hosted_room_policy.publication_exists(
            placement=placement, room_id=ROOM, task_id="t", status="settled", execution_generation=0))


def _publish(seq_now: float, task_id: str) -> Seed:
    """A terminal publication receipt for ``task_id``, as the policy checkpoint records it."""
    return Seed(lambda path: raw(
        path, "INSERT INTO hosted_room_policy_publications (room_id, task_id, kind, execution_generation, seq)"
              " VALUES (?, ?, 'turn.settled', 0, ?)", (ROOM, task_id, int(seq_now))))


def test_prune_published_terminal_tasks_by_count_and_age(hosted, tmp_path, legacy_store):
    run_chain(hosted, tmp_path, "profile", [
        _lease(now=100.0, ttl=1000.0, save="lease"),
        *(_admit(n, now=100.0 + n) for n in (1, 2, 3, 4)),
        *(step for n in (1, 2, 3) for step in (
            _start(n, "lease", now=110.0 + n, save=f"attempt{n}"),
            _settle(f"attempt{n}", now=120.0 + n, settlement=f"settle-{n}"))),
        _checkpoint(),
        _publish(1, "task-1"),
        _publish(2, "task-2"),
        Step("prune_published_terminal_tasks", kwargs=dict(room_id=ROOM, retention_seconds=50.0, retain=10), now=170.0),
        Step("prune_published_terminal_tasks", kwargs=dict(room_id=ROOM, retention_seconds=50.0, retain=10), now=172.0),
        _publish(3, "task-3"),
        Step("prune_published_terminal_tasks", kwargs=dict(room_id=ROOM, retain=0), now=173.0),
        Step("list_tasks", kwargs=dict(room_id=ROOM)),
    ])


def test_work_record_refusals_escape_and_roll_the_transition_back(hosted, tmp_path, legacy_store):
    # CONTRACT: the lineage read of a room past epoch 1 and the roster check raise out of the capture and roll the
    # transition back; a disbanded room is unavailable to the driver.
    run_chain(hosted, tmp_path, "profile", [
        _admit(1, now=101.0),
        Seed(lambda path: raw(path, "UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id=?", (ROOM,))),
        _admit(2, now=102.0, expect="AuthorityHistoryError"),
        _lease(now=103.0, epoch=2, save="lease"),
        Seed(lambda path: raw(path, "UPDATE hosted_rooms SET authority_epoch=1, members_json='{}' WHERE room_id=?",
                              (ROOM,))),
        _admit(2, now=104.0, expect="HostedRoomError"),
        Seed(lambda path: raw(path, "UPDATE hosted_rooms SET members_json=? WHERE room_id=?",
                              ('[{"member_id":"m1","profile":"p1"}]', ROOM))),
        _admit(2, now=105.0),
        Seed(lambda path: legacy_import("gateway.hosted_rooms").disband_room(
                 path, room_id=ROOM, expected_gateway_id=GATEWAY, expected_epoch=1, now=106.0),
             lambda hosted, placement: hosted.store.hosted_rooms.disband_room(
                 placement=placement, room_id=ROOM, expected_gateway_id=GATEWAY, expected_epoch=1, now=106.0)),
        _admit(3, now=107.0, expect="RoomUnavailableError"),
        _lease(now=108.0, expect="RoomUnavailableError"),
        Step("cancel_task", (Ident(2),), dict(cancel_id="cancel-2", expected_cancel_generation=0), now=109.0),
    ])


def test_backend_refusals_are_the_contract_classes(hosted, m1):
    hosted.prepare("profile")
    seed_room(hosted, "profile")
    repo = SQLiteHostedRoomDriverRepository(hosted.store)
    identity = contract.HostedTaskIdentity(ROOM, "task-1", "thread-1", "turn-1")
    with pytest.raises(contract.TaskConflictError, match="^task does not exist$") as caught:
        repo.get_task(identity, placement="profile")
    assert isinstance(caught.value, contract.DriverStateError) and not isinstance(caught.value, m1.DriverStateError)
    with pytest.raises(contract.DriverValidationError, match="^missing payload fields: prompt$"):
        repo.admit_task(identity, placement="profile", payload={"target_profile": "p1", "source_event_seq": 1}, now=1.0)


def test_driver_schema_matches_m1_and_migrates_the_old_status_check(hosted, tmp_path, m1):
    # CONTRACT: the driver tables appear when the first driver verb opens the file, as main's driver connection creates
    # them; the room verbs never create them.
    path = hosted.prepare("profile")
    seed_room(hosted, "profile")
    hosted.store.hosted_rooms.list_rooms(placement="profile")
    assert schema(path, "*driver*") == []
    legacy = tmp_path / "m1" / "state.db"
    legacy.parent.mkdir()
    m1_room(legacy)
    m1._connect(legacy).close()
    assert SQLiteHostedRoomDriverRepository(hosted.store).list_tasks(placement="profile", room_id=ROOM) == ()
    assert schema(path, "*driver*") == schema(legacy, "*driver*")

    # A draft task table whose status CHECK predates 'deferred' and 'stopping' is rebuilt with its rows.
    hosted_old = make_hosted("profile-old")
    try:
        old_path = hosted_old.path("profile")
        seed_room(hosted_old, "profile")
        draft = schema(legacy, "*hosted_room_driver_*")
        for kind, _name, _table, sql in draft:
            if kind == "table":
                raw(old_path, sql.replace(", 'deferred', 'stopping'", ""))
        for kind, _name, _table, sql in draft:
            if kind == "index":
                raw(old_path, sql)
        raw(old_path, "INSERT INTO hosted_room_driver_tasks (room_id, task_id, thread_id, turn_id, source_event_seq,"
                      " payload_json, payload_digest, status, created_at, updated_at) VALUES"
                      " (?, 'task-1', 'thread-1', 'turn-1', 1, ?, ?, 'queued', 1.0, 1.0)",
            (ROOM, *m1._task_payload(_payload(1))[1:]))
        old_twin = twin(old_path, tmp_path / "old")
        m1._connect(old_twin).close()
        tasks = SQLiteHostedRoomDriverRepository(hosted_old.store).list_tasks(placement="profile", room_id=ROOM)
        assert [task["identity"].task_id for task in tasks] == ["task-1"]
        migrated = schema(old_path, "*driver*")
        assert migrated == schema(old_twin, "*driver*")
        (task_sql,) = [sql for kind, name, _table, sql in migrated if kind == "table" and name == "hosted_room_driver_tasks"]
        assert "'deferred', 'stopping'" in task_sql
        assert dump(old_path, DRIVER_TABLES) == dump(old_twin, DRIVER_TABLES)
    finally:
        hosted_old.backend.close()
