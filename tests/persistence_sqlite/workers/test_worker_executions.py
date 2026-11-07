"""The execution verbs outside the receipt stream (registration, adoption, the claim check, the execution reads, the
admission worker's reservation and loss, the pending delegation event) and the child reservation, each against main's
own function on copies of one seed and each one transaction."""

from __future__ import annotations

import secrets
import sqlite3
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from persistence import ExecutionGeneration
from persistence.workers.executions import WorkerExecutionId
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.authority._support import seed_legacy
from tests.persistence_sqlite.workers._support import (
    EXECUTION, NOW, SECRET, SESSION, frozen, ident, legacy_ledger, one_transaction, parity, restore_seed, rolls_back,
    seed_worker, store_path,
)

__all__ = ["frozen", "legacy_store"]

ADMISSION_PRINCIPAL = "principal-1"


def _public(record: Any) -> dict[str, Any]:
    return {"execution_id": record.execution_id, "session_id": record.session_id, "kind": record.kind,
            "owner_epoch": record.owner_epoch, "generation": record.generation, "status": record.status,
            "last_sequence": record.last_sequence}


def _claimed(path: Any, then: Any) -> int:
    """One session whose admitted input ``epoch`` claimed (runtime generation 1), then ``then(db, epoch)``."""
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    epochs: list[int] = []

    def seed(db: Any) -> None:
        db.create_session(SESSION, source="cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        admission = admit_session_input(db, epoch=epoch, principal_id=ADMISSION_PRINCIPAL, session_id=SESSION,
                                        request_id="r1", payload={"text": "hi"})
        claim_session_input(db, epoch=epoch, session_id=SESSION)
        _CLAIMED["admission_id"] = admission["admission_id"]
        then(db, epoch)
        epochs.append(epoch)

    seed_legacy(path, seed)
    return epochs[0]


_CLAIMED: dict[str, str] = {}


def _registered_then_rotated(db: Any, epoch: int) -> None:
    from hermes_state_runtime import begin_runtime_epoch, register_worker_execution

    register_worker_execution(db, epoch=epoch, execution_id=EXECUTION, session_id=SESSION, generation=1,
                              kind="compute", adoption_secret=SECRET)
    begin_runtime_epoch(db, instance_id="successor")


# ── adopt ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("secret,reason", [(SECRET, None), ("wrong-secret", "permission_denied"), ("", "invalid_params")])
def test_adopt_matches_m1_and_moves_the_linked_admission(backend, scope_a, frozen, legacy_store, secret, reason):
    from hermes_state_runtime import adopt_worker_execution

    path = store_path(scope_a)
    epoch = _claimed(path, _registered_then_rotated) + 1
    store = backend.open(scope_a)
    args = dict(epoch=epoch, execution_id=EXECUTION, session_id=SESSION, generation=1, adoption_secret=secret)
    _before, after = parity(path, lambda db: adopt_worker_execution(db, **args),
                           lambda: store.worker_executions.adopt(**args), _public)
    assert getattr(after[1], "reason", None) == reason
    if reason is None:
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT owner_epoch,status FROM session_admissions").fetchall() == [(epoch, "started")]
        rolls_back(store, path, lambda: store.worker_executions.adopt(**args), "worker_executions", "UPDATE")
        one_transaction(store, lambda: store.worker_executions.adopt(**args))


# ── the claim check and the execution reads ──────────────────────────────────


def _legacy_verify(params: dict[str, Any], claim: str) -> Any:
    from tests.persistence_sqlite._main_callers.session_worker import _verify

    return lambda db: _verify(SimpleNamespace(authority=SimpleNamespace(db=db)), None, params, claim)


@pytest.mark.parametrize("override,reason", [
    ({}, None), ({"execution_id": "ghost"}, "not_found"), ({"session_id": "other"}, "permission_denied"),
    ({"generation": 1}, "stale_generation"), ({"generation": "0"}, "stale_generation"), ({"claim": "nope"}, "permission_denied"),
])
def test_verify_claim_matches_m1(backend, scope_a, frozen, legacy_store, override, reason):
    path = store_path(scope_a)
    seed_worker(path)
    store = backend.open(scope_a)
    params = {"execution_id": EXECUTION, "session_id": SESSION, "generation": 0}
    params.update({k: v for k, v in override.items() if k != "claim"})
    claim = override.get("claim", SECRET)
    verb = partial(store.worker_executions.verify_claim, **params, adoption_secret=claim)
    _before, after = parity(path, _legacy_verify(params, claim), verb, lambda record: None)
    assert getattr(after[1], "reason", None) == reason
    if reason is None:
        assert one_transaction(store, verb, kind="read").last_sequence == 0


def test_execution_reads_match_m1(backend, scope_a, frozen, legacy_store):
    from tests.persistence_sqlite._main_callers.session_managed_worker import _execution_tip

    path = store_path(scope_a)
    seed_worker(path)
    store = backend.open(scope_a)
    workers = store.worker_executions
    for execution_id in (EXECUTION, "ghost", ""):
        parity(path, lambda db: _execution_tip(db, execution_id), lambda: workers.execution_session(execution_id=execution_id))
    parity(path, lambda db: bool(db._read_all("SELECT 1 FROM worker_executions WHERE status != 'terminal' LIMIT 1")),
           workers.has_live_worker)
    assert one_transaction(store, lambda: workers.execution_session(execution_id=EXECUTION), kind="read") == SESSION
    assert one_transaction(store, workers.has_live_worker, kind="read") is True
    assert one_transaction(store, lambda: workers.claim_origin(execution_id=EXECUTION, session_id=SESSION), kind="read") == SESSION
    publish = {"execution_id": EXECUTION, "session_id": SESSION, "generation": ExecutionGeneration(0), "sequence": 1,
               "operation": "compression.publish", "payload": {}}
    assert one_transaction(store, lambda: workers.publication_target(**publish), kind="read") == SESSION
    other = dict(publish, operation="transcript.append")
    holder = store.holder()
    from tests.persistence_sqlite.authority._support import counted_transactions

    with counted_transactions(holder) as counts:  # any other operation answers without reading
        assert workers.publication_target(**other) == SESSION
    assert counts == {"write": 0, "read": 0}


def test_has_live_worker_is_false_once_every_execution_ended(backend, scope_a, frozen, legacy_store):
    from hermes_state_runtime import mutate_worker_execution

    path = store_path(scope_a)
    seed_worker(path, extra=lambda db, epoch: mutate_worker_execution(db, **ident(epoch), operation="execution.finish", payload={}))
    store = backend.open(scope_a)
    parity(path, lambda db: bool(db._read_all("SELECT 1 FROM worker_executions WHERE status != 'terminal' LIMIT 1")),
           store.worker_executions.has_live_worker)
    assert store.worker_executions.has_live_worker() is False


# ── the pending delegation event ─────────────────────────────────────────────

DELEGATION = "deleg-1"


def _completed(db: Any, epoch: int) -> None:
    legacy_ledger("delegation.dispatch", {"delegation_id": DELEGATION, "task": {"goal": "g"}, "dispatched_at": NOW - 9,
                                          "session_key": "", "parent_session_id": SESSION, "origin_ui_session_id": "",
                                          "origin_session_id": ""}, ident(epoch, 1))(db)
    legacy_ledger("delegation.complete", {"event": {"type": "async_delegation", "delegation_id": DELEGATION,
                                                    "status": "completed", "completed_at": NOW - 1,
                                                    "parent_session_id": SESSION},
                                          "result": {"summary": "s"}}, ident(epoch, 2))(db)


@pytest.mark.parametrize("delegation_id,execution_id", [(DELEGATION, EXECUTION), ("ghost", EXECUTION), (DELEGATION, "other"), ("", EXECUTION)])
def test_pending_delegation_event_matches_m1(backend, scope_a, frozen, legacy_store, delegation_id, execution_id):
    import json

    from tests.persistence_sqlite._main_callers.session_worker_delegation import _owned

    def legacy(db: Any) -> Any:
        with db._read_ctx() as conn:
            row = _owned(conn, SESSION, execution_id, delegation_id)
            return json.loads(row["event_json"]) if row["delivery_state"] == "pending" else None

    path = store_path(scope_a)
    seed_worker(path, extra=_completed)
    store = backend.open(scope_a)
    verb = partial(store.worker_executions.pending_delegation_event, execution_id=execution_id, session_id=SESSION,
                   delegation_id=delegation_id)
    before, after = parity(path, legacy, verb)
    if before[0] == "ok":
        assert after[1]["delegation_id"] == DELEGATION
        one_transaction(store, verb, kind="read")


# ── the admission worker's loss ──────────────────────────────────────────────

PID, BIRTH, RESERVED_SECRET = 4242, 123.5, "r" * 43


def _reserved(db: Any, epoch: int) -> None:
    """The claimed admission's private worker, reserved through main's own reservation."""
    import secrets as secrets_module

    from tests.persistence_sqlite._main_callers import session_worker_reservation as reservation

    authority = SimpleNamespace(db=db, epoch=epoch, profile_id="profile-a", _require_admission_open=lambda: None)
    original_verify, original_token = reservation.verify_worker_identity, secrets_module.token_urlsafe
    reservation.verify_worker_identity = lambda process, hello: (PID, BIRTH)
    secrets_module.token_urlsafe = lambda n: RESERVED_SECRET
    try:
        reservation.reserve_admission_worker(authority, admission_id=_CLAIMED["admission_id"], process=None,
                                             principal_id=ADMISSION_PRINCIPAL, hello=None)
    finally:
        reservation.verify_worker_identity, secrets_module.token_urlsafe = original_verify, original_token


def test_lose_reservation_matches_m1(backend, scope_a, frozen, legacy_store):
    from tests.persistence_sqlite._main_callers.session_worker_reservation import lose_admission_worker

    path = store_path(scope_a)
    epoch = _claimed(path, _reserved)
    store = backend.open(scope_a)
    admission_id = _CLAIMED["admission_id"]
    execution_id = "admission-worker:" + admission_id
    scope = {"epoch": epoch, "execution_id": execution_id, "generation": 1}
    args = dict(epoch=epoch, admission_id=admission_id, execution_id=WorkerExecutionId(execution_id),
                generation=ExecutionGeneration(1))
    authority = SimpleNamespace(db=None)

    def legacy(db: Any) -> Any:
        authority.db = db
        return lose_admission_worker(authority, {"admission_id": admission_id}, scope)

    before, after = parity(path, legacy, lambda: store.worker_executions.lose_reservation(**args))
    assert before == ("ok", None)
    restore_seed(path)
    rolls_back(store, path, lambda: store.worker_executions.lose_reservation(**args), "sessions", "UPDATE")
    one_transaction(store, lambda: store.worker_executions.lose_reservation(**args))
    restore_seed(path)
    for stale in (dict(args, generation=ExecutionGeneration(2)), dict(args, execution_id=WorkerExecutionId("ghost"))):
        legacy_scope = dict(scope, generation=stale["generation"], execution_id=stale["execution_id"])
        before, after = parity(path, lambda db: lose_admission_worker(SimpleNamespace(db=db), {"admission_id": admission_id}, legacy_scope),
                               lambda: store.worker_executions.lose_reservation(**stale))
        assert after[1].reason == "stale_generation"


# ── the child reservation's refusal receipts ─────────────────────────────────


@pytest.mark.parametrize("refused,limits,code", [
    ("runtime_draining", (10, 1), "runtime_draining"), (None, (0, 1), "delegation_capacity"),
    (None, (10, 0), "delegation_depth_exceeded"),
])
def test_reserve_child_refusals_are_receipts_like_m1(backend, scope_a, frozen, legacy_store, refused, limits, code):
    from tests.persistence_sqlite._main_callers.session_worker_child import reserve_child_execution
    from hermes_state_runtime import apply_worker_receipt

    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(epoch)
    identity = {"profile_id": "profile-a", "pid": PID, "birth": BIRTH}
    secret = "s" * 24

    def legacy(db: Any) -> Any:
        handler = partial(reserve_child_execution, epoch=epoch, execution_id=EXECUTION, identity=identity,
                          principal="principal", limits=limits, refused=refused)
        return apply_worker_receipt(db, **idn, operation="child.reserve", payload={"secret": secret},
                                    handlers={"child.reserve": handler})

    def verb() -> Any:
        return store.worker_executions.reserve_child(**idn, secret=secret, **identity, principal="principal",
                                                     max_concurrent_children=limits[0], max_spawn_depth=limits[1],
                                                     refused=refused)

    _before, after = parity(path, legacy, verb, lambda reservation: {"refused": reservation.refused})
    assert after[1].refused == code and after[1].child is None
    restore_seed(path)
    rolls_back(store, path, verb)
    one_transaction(store, verb)


def test_reserve_child_refuses_a_bad_secret_like_m1(backend, scope_a, frozen, legacy_store):
    from tests.persistence_sqlite._main_callers.session_worker_child import reserve_child_execution
    from hermes_state_runtime import apply_worker_receipt

    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(epoch)
    for secret in ("short", "s" * 257):
        handler = partial(reserve_child_execution, epoch=epoch, execution_id=EXECUTION, identity={}, principal="p",
                          limits=(10, 1), refused=None)
        _before, after = parity(
            path, lambda db: apply_worker_receipt(db, **idn, operation="child.reserve", payload={"secret": secret},
                                                  handlers={"child.reserve": handler}),
            lambda: store.worker_executions.reserve_child(**idn, secret=secret, profile_id="p", pid=1, birth=1.0, principal="p",
                                                          max_concurrent_children=10, max_spawn_depth=1))
        assert after[1].reason == "invalid_params"


def test_reserve_child_matches_m1(backend, scope_a, frozen, legacy_store, monkeypatch):
    import hermes_state_ids
    import session_rules.session_ids

    from tests.persistence_sqlite._main_callers.session_worker_child import reserve_child_execution
    from hermes_state_runtime import apply_worker_receipt

    monkeypatch.setattr(hermes_state_ids, "new_session_id", lambda *a, **k: "child-session")
    monkeypatch.setattr(session_rules.session_ids, "new_session_id", lambda *a, **k: "child-session")
    monkeypatch.setattr(secrets, "token_hex", lambda n: "0123456789abcdef")
    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(epoch)
    identity = {"profile_id": "profile-a", "pid": 4242, "birth": 12.5}
    secret = "s" * 24

    def legacy(db: Any) -> Any:
        handler = partial(reserve_child_execution, epoch=epoch, execution_id=EXECUTION, identity=identity,
                          principal="principal", limits=(10, 1), refused=None)
        return apply_worker_receipt(db, **idn, operation="child.reserve", payload={"secret": secret},
                                    handlers={"child.reserve": handler})

    def verb() -> Any:
        return store.worker_executions.reserve_child(**idn, secret=secret, **identity, principal="principal",
                                                     max_concurrent_children=10, max_spawn_depth=1)

    _before, after = parity(path, legacy, verb, lambda r: {"child": {"session_id": r.child.session_id,
                                                                    "execution_id": r.child.execution_id,
                                                                    "generation": r.child.generation}})
    assert after[1].child.execution_id == EXECUTION + "/child:0123456789abcdef"
    restore_seed(path)
    rolls_back(store, path, verb)
    one_transaction(store, verb)


# ── registration ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("require_idle,kind,secret,reason", [
    (False, "compute", SECRET, None), (True, "cron", SECRET, None), (False, "bogus", SECRET, "invalid_params"),
    (False, "compute", "", "invalid_params"),
])
def test_register_matches_m1(backend, scope_a, frozen, legacy_store, require_idle, kind, secret, reason):
    from hermes_state_runtime import register_worker_execution

    path = store_path(scope_a)
    epoch = seed_worker(path, extra=lambda db, epoch: db.create_session("fresh", source="cli"))
    store = backend.open(scope_a)
    args = dict(epoch=epoch, execution_id="exec-2", session_id="fresh", generation=0, kind=kind, adoption_secret=secret,
                require_idle=require_idle)
    _before, after = parity(path, lambda db: register_worker_execution(db, **args), lambda: store.worker_executions.register(**args),
                           lambda r: {"execution_id": r.execution_id, "session_id": r.session_id, "kind": r.kind,
                                      "owner_epoch": r.owner_epoch, "generation": r.generation, "status": r.status,
                                      "last_sequence": r.last_sequence})
    assert getattr(after[1], "reason", None) == reason
    if reason is None:
        restore_seed(path)
        rolls_back(store, path, lambda: store.worker_executions.register(**args), "worker_executions", "INSERT")
        one_transaction(store, lambda: store.worker_executions.register(**args))
        # The same registration again returns the row; another digest on it is a conflict.
        assert store.worker_executions.register(**args).status == "registered"


def test_register_refuses_a_session_holding_a_live_worker_like_m1(backend, scope_a, frozen, legacy_store):
    from hermes_state_runtime import register_worker_execution

    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    args = dict(epoch=epoch, execution_id="exec-2", session_id=SESSION, generation=0, kind="compute", adoption_secret=SECRET)
    _before, after = parity(path, lambda db: register_worker_execution(db, **args), lambda: store.worker_executions.register(**args))
    assert after[1].reason == "stale_generation"


# ── the admission worker's reservation ───────────────────────────────────────


def test_reserve_for_admission_matches_m1(backend, scope_a, frozen, legacy_store):
    from tests.persistence_sqlite._main_callers import session_worker_reservation as reservation
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    path = store_path(scope_a)
    found: dict[str, Any] = {}

    def seed(db: Any) -> None:
        db.create_session(SESSION, source="cli")
        found["epoch"] = epoch = begin_runtime_epoch(db, instance_id="owner")
        found["admission"] = admit_session_input(db, epoch=epoch, principal_id="principal-1", session_id=SESSION,
                                                 request_id="r1", payload={"text": "hi"})["admission_id"]
        claim_session_input(db, epoch=epoch, session_id=SESSION)

    seed_legacy(path, seed)
    store = backend.open(scope_a)
    epoch, admission_id = found["epoch"], found["admission"]
    args = dict(epoch=epoch, admission_id=admission_id, principal_id="principal-1", profile_id="profile-a", pid=4242,
                birth=12.5, secret="r" * 43, owner=None)

    def legacy(db: Any) -> Any:
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id="profile-a", _require_admission_open=lambda: None)
        original_verify, original_token = reservation.verify_worker_identity, secrets.token_urlsafe
        reservation.verify_worker_identity = lambda process, hello: (4242, 12.5)
        secrets.token_urlsafe = lambda n: "r" * 43
        try:
            return reservation.reserve_admission_worker(authority, admission_id=admission_id, process=None,
                                                        principal_id="principal-1", hello=None)
        finally:
            reservation.verify_worker_identity, secrets.token_urlsafe = original_verify, original_token

    before, _after = parity(path, legacy, lambda: store.worker_executions.reserve_for_admission(**args),
                           lambda scope: {"profile_id": scope.profile_id, "session_id": scope.session_id,
                                          "execution_id": scope.execution_id, "generation": scope.generation,
                                          "pid": scope.pid, "birth": scope.birth, "secret": scope.secret, "epoch": scope.epoch})
    assert before[0] == "ok"
    restore_seed(path)
    rolls_back(store, path, lambda: store.worker_executions.reserve_for_admission(**args), "worker_executions", "INSERT")
    one_transaction(store, lambda: store.worker_executions.reserve_for_admission(**args))
    for wrong, reason in ((dict(args, principal_id="other"), "permission_denied"), (dict(args, epoch=epoch + 1), "stale_epoch")):
        restore_seed(path)
        with pytest.raises(Exception) as caught:
            store.worker_executions.reserve_for_admission(**wrong)
        assert caught.value.reason == reason
