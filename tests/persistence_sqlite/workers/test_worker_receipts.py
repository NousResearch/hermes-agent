"""The worker receipt transaction against main: the pre-store checks, the fences in main's order (epoch, assignment,
owner epoch, stored receipt, live status, strict sequence), exact replay before the terminal check, the revision bump
only when rows changed, the stored receipt bytes over a whole stream, the retired operations, and the typed verbs'
key-set check before the fences."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from persistence import ExecutionGeneration, SessionId
from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._runtime import refused
from persistence.sqlite.workers import _gateway, executions
from persistence.sqlite.workers._receipt import mutate_worker_execution
from persistence.workers.compaction import WorkerCompaction, WorkerCompressionPublication
from persistence.workers.executions import WorkerExecutionId, WorkerTurnLease
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.authority._support import assert_same, dump, run_both
from tests.persistence_sqlite.workers._support import (
    EXECUTION, SESSION, WORKER_TABLES, execution, frozen, ident, legacy_persist, parity, revision, seed_store,
    seed_worker, store_path,
)

__all__ = ["frozen", "legacy_store"]

FALLBACK = ("compression.fallback", {"value": 3})


def _fallback(store: Any, idn: dict[str, Any], streak: int = 3) -> Any:
    return store.worker_executions.set_compression_fallback_streak(**idn, streak=streak)


def _refused(result: tuple[str, Any], reason: str) -> None:
    """An ``AuthorityRefusal`` with ``reason``."""
    assert result[0] == "error" and getattr(result[1], "reason", None) == reason, result


def _raw(db: Any, sql: str, params: tuple = ()) -> None:
    db._execute_write(lambda conn: conn.execute(sql, params))


# ── pre-store checks ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("sequence", [0, -1, True, 1.0, "1"])
def test_a_bad_sequence_is_invalid_params_before_the_epoch_fence(backend, scope_a, frozen, sequence):
    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    stale = ident(epoch + 7, sequence)
    _before, after = parity(path, legacy_persist(*FALLBACK, stale), lambda: _fallback(store, stale))
    _refused(after, "invalid_params")


def test_an_oversized_payload_is_invalid_params_before_the_epoch_fence(backend, scope_a, frozen):
    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    stale, prompt = ident(epoch + 7), "x" * (4 * 1024 * 1024 + 1)
    _before, after = parity(path, legacy_persist("session.prompt", {"system_prompt": prompt}, stale),
                           lambda: store.worker_executions.set_session_system_prompt(**stale, system_prompt=prompt))
    _refused(after, "invalid_params")


# ── fences in main's order ───────────────────────────────────────────────────


def _other_session(db: Any, epoch: int) -> None:
    db.create_session("other", source="cli")


def _moved_generation(db: Any, epoch: int) -> None:
    _raw(db, "UPDATE sessions SET runtime_generation=1 WHERE id=?", (SESSION,))


def _new_epoch(db: Any, epoch: int) -> None:
    from hermes_state_runtime import begin_runtime_epoch

    begin_runtime_epoch(db, instance_id="successor")


def _finished(db: Any, epoch: int) -> None:
    legacy_persist(*FALLBACK, ident(epoch, 1))(db)
    legacy_persist("execution.finish", {}, ident(epoch, 2))(db)


def _written(db: Any, epoch: int) -> None:
    legacy_persist(*FALLBACK, ident(epoch, 1))(db)


FENCES = [
    # (name, seed, ident overrides as (epoch offset, sequence, extra), reason)
    ("epoch_before_assignment", None, (5, 1, {"execution_id": "ghost"}), "stale_epoch"),
    ("unknown_execution", None, (0, 1, {"execution_id": "ghost"}), "not_found"),
    ("another_session", _other_session, (0, 1, {"session_id": "other"}), "permission_denied"),
    ("row_generation", None, (0, 1, {"generation": 1}), "stale_generation"),
    ("session_generation", _moved_generation, (0, 1, {"generation": 0}), "stale_generation"),
    ("caller_epoch_after_rotation", _new_epoch, (0, 1, {}), "stale_epoch"),
    ("owner_epoch_after_rotation", _new_epoch, (1, 1, {}), "stale_epoch"),
    ("conflicting_replay_before_status", _finished, (0, 1, {"streak": 4}), "admission_conflict"),
    ("terminal_after_replay", _finished, (0, 3, {}), "stale_generation"),
    ("strict_sequence", _written, (0, 3, {}), "invalid_params"),
    ("generation_type", None, (0, 1, {"generation": "0"}), "stale_generation"),
]


@pytest.mark.parametrize("name,seed,overrides,reason", FENCES, ids=[f[0] for f in FENCES])
def test_fences_answer_in_m1_order(backend, scope_a, frozen, name, seed, overrides, reason):
    path = store_path(scope_a)
    epoch = seed_worker(path, extra=seed)
    store = backend.open(scope_a)
    offset, sequence, extra = overrides
    extra = dict(extra)
    streak = extra.pop("streak", 3)
    idn = ident(epoch + offset, sequence, **extra)
    before, after = parity(path, legacy_persist("compression.fallback", {"value": streak}, idn),
                           lambda: _fallback(store, idn, streak))
    _refused(before, reason)
    _refused(after, reason)


def test_an_exact_replay_returns_the_stored_result_even_after_the_execution_ended(backend, scope_a, frozen):
    path = store_path(scope_a)
    epoch = seed_worker(path, extra=_finished)
    store = backend.open(scope_a)
    before, _after = parity(path, legacy_persist(*FALLBACK, ident(epoch, 1)), lambda: _fallback(store, ident(epoch, 1)),
                           wire=lambda value: {"value": value})
    assert before == ("ok", {"value": None})
    assert execution(store).last_sequence == 2


def test_a_retired_read_receipt_is_stale_generation_after_the_execution_ended(backend, scope_a, frozen):
    def read_then_finish(db: Any, epoch: int) -> None:
        legacy_persist("compression.context", {"target": SESSION}, ident(epoch, 1))(db)
        legacy_persist("execution.finish", {}, ident(epoch, 2))(db)

    path = store_path(scope_a)
    epoch = seed_worker(path, extra=read_then_finish)
    store = backend.open(scope_a)
    idn = ident(epoch, 1)
    _before, after = parity(path, legacy_persist("compression.context", {"target": SESSION}, idn),
                           lambda: store.worker_executions.compression_session_context(**idn, target_session_id=SESSION))
    _refused(after, "stale_generation")


# ── a publication's assignment ───────────────────────────────────────────────

CHILD = "sess-worker-child"
LOCK = "pid=1:compress"
PUBLICATION = WorkerCompressionPublication(child_session_id=SessionId(CHILD), source="cli",
                                           messages=[{"role": "user", "content": "carried"}], compression_lock_holder=LOCK)


def _published(db: Any, epoch: int) -> None:
    db.try_acquire_compression_lock(SESSION, LOCK, ttl_seconds=600)
    legacy_persist("compression.publish", PUBLICATION.payload(), ident(epoch, 1))(db)


def test_an_identical_publication_replay_addressed_to_the_parent_resolves_the_recorded_move(backend, scope_a, frozen):
    path = store_path(scope_a)
    epoch = seed_worker(path, extra=_published)
    store = backend.open(scope_a)
    idn = ident(epoch, 1)
    _before, after = parity(path, legacy_persist("compression.publish", PUBLICATION.payload(), idn),
                           lambda: store.worker_executions.publish_compression_child(**idn, publication=PUBLICATION),
                           wire=lambda r: {"value": None, "worker_assignment": {"parent": r.parent, "session_id": r.session_id},
                                           "annotations": [a.as_wire() for a in r.annotations]})
    assert after[1].session_id == CHILD
    # A different publication at the same sequence is addressed to the parent the worker left.
    other = WorkerCompressionPublication(child_session_id=SessionId("elsewhere"), source="cli", messages=[],
                                         compression_lock_holder=LOCK)
    with pytest.raises(AuthorityRefusal) as caught:
        store.worker_executions.publish_compression_child(**idn, publication=other)
    assert caught.value.reason == "permission_denied"


def test_receipt_reads_follow_the_recorded_move(backend, scope_a, frozen, legacy_store):
    from hermes_state_worker_compression import worker_claim_origin, worker_retry_target

    path = store_path(scope_a)
    seed_worker(path, extra=_published)
    store = backend.open(scope_a)
    workers = store.worker_executions
    params = {"execution_id": EXECUTION, "session_id": SESSION, "generation": 0, "sequence": 1,
              "operation": "compression.publish", "payload": PUBLICATION.payload()}
    for legacy, verb in (
        (lambda db: worker_claim_origin(db, params), lambda: workers.claim_origin(execution_id=EXECUTION, session_id=SESSION)),
        (lambda db: worker_retry_target(db, params), lambda: workers.publication_target(**params)),
        (lambda db: worker_retry_target(db, dict(params, operation="compression.archive")),
         lambda: workers.publication_target(**dict(params, operation="compression.archive"))),
        (lambda db: worker_retry_target(db, dict(params, payload={})), lambda: workers.publication_target(**dict(params, payload={}))),
        (lambda db: worker_claim_origin(db, dict(params, execution_id="ghost")),
         lambda: workers.claim_origin(execution_id=WorkerExecutionId("ghost"), session_id=SESSION)),
    ):
        parity(path, legacy, verb)
    assert workers.claim_origin(execution_id=EXECUTION, session_id=CHILD) == SESSION
    assert workers.publication_target(**params) == CHILD
    assert workers.execution_session(execution_id=EXECUTION) == CHILD


# ── revision ─────────────────────────────────────────────────────────────────


def test_the_revision_moves_only_when_the_operation_wrote(backend, scope_a, frozen):
    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store)
    workers = store.worker_executions
    start = revision(path)
    sequence = iter(range(1, 100))

    def idn() -> dict[str, Any]:
        return ident(epoch, next(sequence))

    assert workers.compression_watermark(**idn()) == 0
    assert workers.compression_lock_holder(**idn()) is None
    assert workers.latest_conversation_role(**idn()) is None
    assert workers.renew_compression_lock(**idn(), lease=WorkerTurnLease(holder="nobody")) is False
    assert revision(path) == start
    workers.set_compression_fallback_streak(**idn(), streak=0)  # matches the row though the value stays 0
    assert revision(path) == start + 1
    workers.set_compression_fallback_streak(**idn(), streak=2)
    assert revision(path) == start + 2
    workers.finish(**idn())
    assert revision(path) == start + 3


def test_a_whole_receipt_stream_stores_m1_bytes(backend, scope_a, frozen, legacy_store):
    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    workers = store.worker_executions
    stream = [
        ("compression.watermark", {}, lambda i: workers.compression_watermark(**i)),
        ("compression.fallback", {"value": 1}, lambda i: workers.set_compression_fallback_streak(**i, streak=1)),
        ("compression.recovery", {"value": None}, lambda i: workers.set_compression_recovery_deadline(**i, deadline=None)),
        ("session.title_source", {"source": "llm"}, lambda i: workers.set_session_title_source(**i, source="llm")),
        ("compression.lock.holder", {}, lambda i: workers.compression_lock_holder(**i)),
        ("execution.finish", {}, lambda i: workers.finish(**i)),
    ]

    def legacy(db: Any) -> None:
        for sequence, (operation, payload, _verb) in enumerate(stream, 1):
            legacy_persist(operation, payload, ident(epoch, sequence))(db)

    def verbs() -> None:
        for sequence, (_operation, _payload, verb) in enumerate(stream, 1):
            verb(ident(epoch, sequence))

    before, after = run_both(path, legacy, verbs, WORKER_TABLES)
    assert before[0] == after[0] == "ok", (before, after)
    receipts = dump(path, ("worker_receipts",))["worker_receipts"]
    assert [row[1] for row in receipts] == list(range(1, len(stream) + 1))


# ── retired operations and the typed key-set check ───────────────────────────

RETIRED = ("session.context", "turn.release", "compression.lock.release")


@pytest.mark.parametrize("operation", RETIRED)
def test_a_retired_operation_has_no_handler(backend, scope_a, frozen, operation):
    path = store_path(scope_a)
    epoch = seed_store(backend.open(scope_a))
    payload = {"holder": "h"} if operation != "session.context" else {}
    holder = backend.open(scope_a).holder()
    with pytest.raises(AuthorityRefusal) as caught, refused():
        mutate_worker_execution(holder, **ident(epoch, 1), operation=operation, payload=payload)
    assert caught.value.reason == "invalid_params"
    with pytest.raises(AuthorityRefusal) as caught:
        backend.open(scope_a).worker_executions.persist_operation(**ident(epoch, 1), operation=operation, payload=payload)
    assert caught.value.reason == "invalid_params"
    assert dump(path, ("worker_receipts",))["worker_receipts"] == []


class _Sent(Exception):
    pass


def test_no_verb_sends_a_retired_operation(backend, scope_a, monkeypatch):
    sent: list[str] = []

    def record(*args: Any, operation: str = "child.reserve", **kwargs: Any) -> Any:
        sent.append(operation)
        raise _Sent

    monkeypatch.setattr(executions, "mutate_worker_execution", record)
    monkeypatch.setattr(_gateway, "persist_worker_delegation", record)
    monkeypatch.setattr(_gateway, "reserve_child", record)
    workers = backend.open(scope_a).worker_executions
    receipted = [name for name, fn in inspect.getmembers(type(workers), inspect.isfunction)
                 if not name.startswith("_") and name not in ("publication_target", "persist_operation")
                 and "sequence" in inspect.signature(fn).parameters]
    for name in receipted:
        kwargs = {key: _dummy(key, param) for key, param in inspect.signature(getattr(type(workers), name)).parameters.items()
                  if key != "self" and param.kind is param.KEYWORD_ONLY and param.default is param.empty}
        if name == "create_session":
            kwargs["source"] = "cli"
        with pytest.raises(_Sent):
            getattr(workers, name)(**kwargs)
    # Every receipted verb sends its own operation, and none sends a retired one.
    assert len(receipted) == len(sent) == len(set(sent)) == 54
    assert not set(sent) & set(RETIRED)


def _dummy(name: str, param: inspect.Parameter) -> Any:
    from persistence.conversation.messages import ConversationPeer
    from persistence.workers.compression import WorkerCooldown, WorkerCooldownSnapshot
    from persistence.workers.delegations import (
        WorkerDelegationChild, WorkerDelegationClaim, WorkerDelegationCompletion, WorkerDelegationDispatch,
    )
    from persistence.workers.sessions import WorkerActivity, WorkerBillingRoute

    values = {
        "epoch": 1, "execution_id": EXECUTION, "session_id": SESSION, "generation": ExecutionGeneration(0), "sequence": 1,
        "lease": WorkerTurnLease(holder="h"), "target_session_id": SESSION, "holder": "h", "title": "t", "source": "user",
        "base_title": "t", "activity": WorkerActivity(ts=None, description=None, provenance=None),
        "route": WorkerBillingRoute(provider=None, base_url=None, billing_mode=None), "content": "c", "api_content": "a",
        "row_id": 1, "end_reason": "r", "system_prompt": None, "patch": {}, "tool_pin": None, "messages": [],
        "cooldown": WorkerCooldown(cooldown_until=1.0), "snapshot": WorkerCooldownSnapshot(session_exists=True),
        "streak": 0, "count": 0, "deadline": None, "peer": ConversationPeer(session_key="k", source="cli"),
        "include_ancestors": False, "include_inactive": False, "repair_alternation": False, "include_row_ids": False,
        "include_compacted": False, "compaction": WorkerCompaction(messages=[]), "publication": PUBLICATION,
        "compression_lock_holder": None, "turn_lease_holder": None, "turn_lease_ttl_seconds": 300.0,
        "secret": "s" * 16, "profile_id": "p", "pid": 1, "birth": 1.0, "principal": "p", "max_concurrent_children": 1,
        "max_spawn_depth": 1, "task": "t", "worker_pid": 1, "worker_birth": 1.0, "owner_started_at": None, "delegation_id": "d",
        "dispatch": WorkerDelegationDispatch(delegation_id="d", task={}, dispatched_at=1.0, session_key="",
                                             parent_session_id=SESSION, origin_ui_session_id="", origin_session_id=""),
        "completion": WorkerDelegationCompletion(event={}, result={}),
        "child": WorkerDelegationChild(delegation_id="d", entry={}), "claim": WorkerDelegationClaim(delegation_id="d", claim_id="c"),
    }
    return values[name]


KEY_SETS = [
    ("record_usage", {"input_tokens": 1, "bogus": 1}, "usage.main"),
    ("record_auxiliary_usage", {"task": "t", "actual_cost_usd": 1.0}, "usage.auxiliary"),
    ("create_session", {"model": "m"}, "session.create"),
    ("create_session", {"source": "cli", "bogus": "x"}, "session.create"),
    ("append_transcript", {"messages": [], "bogus": None}, "transcript.append"),
]


@pytest.mark.parametrize("verb,payload,operation", KEY_SETS, ids=[f"{v}-{sorted(p)}" for v, p, _o in KEY_SETS])
def test_a_malformed_key_set_is_invalid_params_before_the_fences(backend, scope_a, frozen, legacy_store, verb, payload,
                                                                 operation):
    # Stated difference from main: main builds the receipt first, so a stale epoch answers before the key set.
    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    stale = ident(epoch + 3)
    before, after = run_both(path, legacy_persist(operation, payload, stale),
                             lambda: getattr(store.worker_executions, verb)(**stale, **payload), WORKER_TABLES)
    _refused(before, "stale_epoch")
    _refused(after, "invalid_params")
    # With the live epoch both refuse the key set.
    live = ident(epoch)
    before, after = run_both(path, legacy_persist(operation, payload, live),
                             lambda: getattr(store.worker_executions, verb)(**live, **payload), WORKER_TABLES)
    assert_same(before, after)
    _refused(after, "invalid_params")
