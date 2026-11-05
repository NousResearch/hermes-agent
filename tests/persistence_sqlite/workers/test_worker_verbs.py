"""Per-verb differential cases for the receipted worker verbs (session rows, transcript, titles, usage, compression,
delegation ledger): each verb against main's own function on copies of one seed (outcome, rows, stored receipt digest
and result), and each verb as one transaction that a failure injected after its first statement leaves without a
trace."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from persistence import SessionId
from persistence.conversation.messages import ConversationPeer
from persistence.workers.compaction import WorkerCompaction, WorkerCompressionPublication
from persistence.workers.compression import WorkerCooldown, WorkerCooldownSnapshot
from persistence.workers.delegations import (
    WorkerDelegationChild, WorkerDelegationClaim, WorkerDelegationCompletion, WorkerDelegationDispatch,
)
from persistence.workers.executions import WorkerTurnLease
from persistence.workers.sessions import WorkerActivity, WorkerBillingRoute
from tests.fixtures.storage_seed import seed_messages
from tests.persistence_sqlite.workers._support import (
    EXECUTION, NOW, SESSION, frozen, ident, legacy_ledger, legacy_persist, one_transaction, owner_started_at, parity,
    rolls_back, seed_store, seed_worker, store_path, worker_birth,
)

__all__ = ["frozen"]

DELEGATION = "deleg-1"
LOCK = "pid=1:compress"
CHILD = "sess-worker-child"
TASK = {"goal": "g", "toolsets": ["web"]}


def _dispatch_payload(delegation_id: str = DELEGATION) -> dict[str, Any]:
    return {"delegation_id": delegation_id, "task": TASK, "dispatched_at": NOW - 10, "session_key": "",
            "parent_session_id": SESSION, "origin_ui_session_id": "", "origin_session_id": ""}


def _event(status: str = "completed") -> dict[str, Any]:
    return {"type": "async_delegation", "delegation_id": DELEGATION, "status": status, "completed_at": NOW - 1,
            "session_key": "", "parent_session_id": SESSION, "origin_ui_session_id": "", "origin_session_id": ""}


def _seed_title(db: Any, epoch: int) -> None:
    db.set_session_title(SESSION, "Title")


def _seed_messages(db: Any, epoch: int) -> None:
    db.append_message(SESSION, "user", "hello")
    db.append_message(SESSION, "assistant", "hi")


def _seed_lock(db: Any, epoch: int) -> None:
    db.try_acquire_compression_lock(SESSION, "pid=1:lock", ttl_seconds=60)


def _seed_dispatch(db: Any, epoch: int) -> None:
    legacy_ledger("delegation.dispatch", _dispatch_payload(), ident(epoch, 1))(db)


def _seed_completed(db: Any, epoch: int) -> None:
    _seed_dispatch(db, epoch)
    legacy_ledger("delegation.complete", {"event": _event(), "result": {"summary": "s"}}, ident(epoch, 2))(db)


def _seed_claimed(db: Any, epoch: int) -> None:
    _seed_completed(db, epoch)
    legacy_ledger("delegation.claim", {"delegation_id": DELEGATION, "claim_id": "c1"}, ident(epoch, 3))(db)


def _seed_compression_lock(db: Any, epoch: int) -> None:
    _seed_messages(db, epoch)
    db.try_acquire_compression_lock(SESSION, LOCK, ttl_seconds=600)


def _seed_compressed(db: Any, epoch: int) -> None:
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (NOW - 30, SESSION)))


# The same seeds through the backend.


def _store_title(store: Any, epoch: int) -> None:
    store.sessions.set_title(SESSION, "Title")


def _store_messages(store: Any, epoch: int) -> None:
    seed_messages(store, SESSION, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"})


def _store_lock(store: Any, epoch: int) -> None:
    store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="pid=1:lock", ttl_seconds=60)


def _store_dispatch(store: Any, epoch: int) -> None:
    store.worker_executions.dispatch_delegation(**ident(epoch, 1), dispatch=WorkerDelegationDispatch(**_dispatch_payload()),
                                                worker_pid=os.getpid(), worker_birth=worker_birth(),
                                                owner_started_at=owner_started_at())


def _store_completed(store: Any, epoch: int) -> None:
    _store_dispatch(store, epoch)
    store.worker_executions.complete_delegation(**ident(epoch, 2), completion=WorkerDelegationCompletion(
        event=_event(), result={"summary": "s"}), worker_pid=os.getpid(), worker_birth=worker_birth())


def _store_claimed(store: Any, epoch: int) -> None:
    _store_completed(store, epoch)
    store.worker_executions.claim_delegation_delivery(**ident(epoch, 3), claim=WorkerDelegationClaim(
        delegation_id=DELEGATION, claim_id="c1"), worker_pid=os.getpid(), worker_birth=worker_birth())


def _store_compression_lock(store: Any, epoch: int) -> None:
    _store_messages(store, epoch)
    store.coordination.try_acquire_compression_lock(session_id=SESSION, holder=LOCK, ttl_seconds=600)


def _store_compressed(store: Any, epoch: int) -> None:
    store.holder()._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (NOW - 30, SESSION)))


def _annotated(result: Any) -> dict[str, Any]:
    return {"count": result.count, "annotations": [a.as_wire() for a in result.annotations]}


PUBLICATION = WorkerCompressionPublication(child_session_id=SessionId(CHILD), source="cli",
                                           messages=[{"role": "user", "content": "carried"}], compression_lock_holder=LOCK)
COMPACTION = WorkerCompaction(messages=[{"role": "user", "content": "summary"}])
APPEND = {"messages": [{"role": "user", "content": "x"}], "compression_lock_holder": None, "turn_lease_holder": None,
          "turn_lease_ttl_seconds": 300.0}


@dataclass(frozen=True)
class Case:
    """One verb call: the legacy operation and payload, the verb, the record-to-wire mapping, the seed, the sequence
    the call takes, the table and event a failure is injected at, whether the legacy side is the ledger, the seeded
    session's key, and the seed through the backend."""

    name: str
    operation: str
    payload: Any
    call: Callable[[Any, dict[str, Any]], Any]
    wire: Callable[[Any], Any] = lambda value: {"value": value}
    seed: Callable[[Any, int], None] | None = None
    sequence: int = 1
    ledger: bool = False
    fail_at: tuple[str, str] = ("worker_receipts", "INSERT")
    session_key: str | None = None
    store_seed: Callable[[Any, int], None] | None = None


def _w(verb: str, **kwargs: Any) -> Callable[[Any, dict[str, Any]], Any]:
    return lambda store, idn: getattr(store.worker_executions, verb)(**idn, **kwargs)


def _ledger(verb: str, **kwargs: Any) -> Callable[[Any, dict[str, Any]], Any]:
    """A delegation ledger verb with this process as the verified worker, as ``legacy_ledger`` passes it."""
    return lambda store, idn: getattr(store.worker_executions, verb)(**idn, **kwargs, worker_pid=os.getpid(),
                                                                      worker_birth=worker_birth())


CASES = [
    Case("finish", "execution.finish", {}, _w("finish"), wire=lambda v: {"status": "terminal"} if v is None else v),
    Case("set_session_title_source", "session.title_source", {"source": "user"}, _w("set_session_title_source", source="user"), seed=_seed_title, store_seed=_store_title),
    Case("next_session_title", "session.next_title", {"base_title": "Title"}, _w("next_session_title", base_title="Title"), seed=_seed_title, store_seed=_store_title),
    Case("touch_session_activity", "session.activity", {"ts": NOW - 5, "description": "reading", "provenance": "tool"},
         _w("touch_session_activity", activity=WorkerActivity(ts=NOW - 5, description="reading", provenance="tool"))),
    Case("clear_session_activity", "session.activity_clear", {}, _w("clear_session_activity")),
    Case("set_session_billing_route", "session.billing_route", {"provider": "p", "base_url": "u", "billing_mode": None},
         _w("set_session_billing_route", route=WorkerBillingRoute(provider="p", base_url="u", billing_mode=None))),
    Case("latest_conversation_role", "session.tail_role", {}, _w("latest_conversation_role"), seed=_seed_messages, store_seed=_store_messages),
    Case("message_role", "session.row_role", {"row_id": 1}, _w("message_role", row_id=1), seed=_seed_messages, store_seed=_store_messages),
    Case("renew_compression_lock", "compression.lock.renew", {"holder": "pid=1:lock", "ttl_seconds": 30.0},
         _w("renew_compression_lock", lease=WorkerTurnLease(holder="pid=1:lock", ttl_seconds=30.0)), seed=_seed_lock, store_seed=_store_lock),
    Case("compression_lock_holder", "compression.lock.holder", {}, _w("compression_lock_holder"), seed=_seed_lock, store_seed=_store_lock),
    Case("record_compression_cooldown", "compression.cooldown.record", {"cooldown_until": NOW + 60, "error": "boom"},
         _w("record_compression_cooldown", cooldown=WorkerCooldown(cooldown_until=NOW + 60, error="boom"))),
    Case("restore_compression_cooldown", "compression.cooldown.restore",
         {"snapshot": {"session_exists": True, "cooldown_until": NOW + 5, "error": "e"}},
         _w("restore_compression_cooldown", snapshot=WorkerCooldownSnapshot(session_exists=True, cooldown_until=NOW + 5, error="e"))),
    Case("clear_compression_cooldown", "compression.cooldown.clear", {}, _w("clear_compression_cooldown")),
    Case("set_compression_fallback_streak", "compression.fallback", {"value": 3}, _w("set_compression_fallback_streak", streak=3)),
    Case("set_compression_ineffective_count", "compression.ineffective", {"value": 2}, _w("set_compression_ineffective_count", count=2)),
    Case("set_compression_recovery_deadline", "compression.recovery", {"value": -5.0}, _w("set_compression_recovery_deadline", deadline=-5.0)),
    Case("compression_watermark", "compression.watermark", {}, _w("compression_watermark"), seed=_seed_messages, store_seed=_store_messages),
    Case("dispatch_delegation", "delegation.dispatch", _dispatch_payload(),
         lambda store, idn: store.worker_executions.dispatch_delegation(
             **idn, dispatch=WorkerDelegationDispatch(**_dispatch_payload()), worker_pid=os.getpid(),
             worker_birth=worker_birth(), owner_started_at=owner_started_at()),
         wire=lambda v: {"value": v}, ledger=True),
    Case("complete_delegation", "delegation.complete", {"event": _event(), "result": {"summary": "s"}},
         _ledger("complete_delegation", completion=WorkerDelegationCompletion(event=_event(), result={"summary": "s"})),
         seed=_seed_dispatch, store_seed=_store_dispatch, sequence=2, ledger=True),
    Case("record_delegation_child", "delegation.child", {"delegation_id": DELEGATION, "entry": {"task_index": 0, "status": "ok"}},
         _ledger("record_delegation_child", child=WorkerDelegationChild(delegation_id=DELEGATION, entry={"task_index": 0, "status": "ok"})),
         seed=_seed_dispatch, store_seed=_store_dispatch, sequence=2, ledger=True),
    Case("read_delegation", "delegation.read", {"delegation_id": DELEGATION}, _ledger("read_delegation", delegation_id=DELEGATION),
         wire=lambda v: {"value": {"delegation_id": v.delegation_id, "origin_session": v.origin_session, "state": v.state,
                                   "dispatched_at": v.dispatched_at, "completed_at": v.completed_at,
                                   "delivery_state": v.delivery_state, "delivery_attempts": v.delivery_attempts,
                                   "origin_session_id": v.origin_session_id, "result": v.result}},
         seed=_seed_completed, store_seed=_store_completed, sequence=3, ledger=True),
    Case("claim_delegation_delivery", "delegation.claim", {"delegation_id": DELEGATION, "claim_id": "c1"},
         _ledger("claim_delegation_delivery", claim=WorkerDelegationClaim(delegation_id=DELEGATION, claim_id="c1")),
         seed=_seed_completed, store_seed=_store_completed, sequence=3, ledger=True),
    *(Case(f"{action}_delegation_delivery", f"delegation.{action}", {"delegation_id": DELEGATION, "claim_id": "c1"},
           _ledger(f"{action}_delegation_delivery", claim=WorkerDelegationClaim(delegation_id=DELEGATION, claim_id="c1")),
           seed=_seed_claimed, store_seed=_store_claimed, sequence=4, ledger=True)
      for action in ("ack", "drop", "defer", "release")),
    Case("withdraw_delegation", "delegation.unscheduled", {"delegation_id": DELEGATION}, _ledger("withdraw_delegation", delegation_id=DELEGATION),
         seed=_seed_dispatch, store_seed=_store_dispatch, sequence=2, ledger=True),
    Case("set_session_title", "session.title", {"title": "New title", "source": "user"},
         _w("set_session_title", title="New title", source="user"),
         wire=lambda r: {"value": r.value} if r.error is None else {"value": r.value, "error": r.error, "title_candidate": r.title_candidate}),
    Case("set_latest_user_api_content", "session.api_content", {"content": "hello", "api_content": "api"},
         _w("set_latest_user_api_content", content="hello", api_content="api"), seed=_seed_messages, store_seed=_store_messages),
    Case("set_message_api_content", "session.message_api_content", {"row_id": 1, "content": "hello", "api_content": "api"},
         _w("set_message_api_content", row_id=1, content="hello", api_content="api"), seed=_seed_messages, store_seed=_store_messages),
    Case("set_user_message_content", "session.user_content", {"row_id": 1, "content": "edited"},
         _w("set_user_message_content", row_id=1, content="edited"), seed=_seed_messages, store_seed=_store_messages),
    Case("create_session", "session.create", {"source": "cli", "model": "m"}, _w("create_session", source="cli", model="m")),
    Case("end_session", "session.end", {"end_reason": "user_exit"}, _w("end_session", end_reason="user_exit")),
    Case("session_lifecycle", "session.lifecycle", {}, _w("session_lifecycle"), wire=lambda r: {"value": r.value},
         seed=_seed_messages, store_seed=_store_messages),
    Case("set_session_system_prompt", "session.prompt", {"system_prompt": "sys"}, _w("set_session_system_prompt", system_prompt="sys")),
    Case("patch_session_sidecars", "session.sidecars", {"patch": {"_usage_anchor": 5}}, _w("patch_session_sidecars", patch={"_usage_anchor": 5})),
    Case("set_session_tool_pin", "session.tools", {"tool_names": {"version": "1", "tools": ["a"]}},
         _w("set_session_tool_pin", tool_pin={"version": "1", "tools": ["a"]})),
    Case("append_transcript", "transcript.append", {"messages": [{"role": "user", "content": "x", "timestamp": NOW - 1}]},
         _w("append_transcript", messages=[{"role": "user", "content": "x", "timestamp": NOW - 1}]), wire=_annotated),
    Case("record_usage", "usage.main", {"input_tokens": 5, "output_tokens": 2, "model": "m", "source": "gui"},
         _w("record_usage", input_tokens=5, output_tokens=2, model="m", source="gui")),
    Case("record_auxiliary_usage", "usage.auxiliary", {"input_tokens": 3, "task": "title"},
         _w("record_auxiliary_usage", task="title", input_tokens=3)),
    Case("reopen_orphaned_compression", "compression.reopen", {}, _w("reopen_orphaned_compression"), seed=_seed_compressed, store_seed=_store_compressed),
    Case("compression_session_context", "compression.context", {"target": SESSION},
         _w("compression_session_context", target_session_id=SESSION), wire=lambda r: {"session": dict(r)}),
    Case("compression_lineage_view", "compression.lineage", {"target": SESSION},
         _w("compression_lineage_view", target_session_id=SESSION),
         wire=lambda r: {"readable_ids": list(r.readable_ids), "lineage": list(r.lineage), "root": r.root, "tip": r.tip,
                         "resume": r.resume, "identity": r.identity}),
    Case("compression_history", "compression.history",
         {"target": SESSION, "include_ancestors": True, "include_inactive": False, "repair_alternation": True,
          "include_row_ids": True, "include_compacted": False},
         _w("compression_history", target_session_id=SESSION, include_ancestors=True, include_inactive=False,
            repair_alternation=True, include_row_ids=True, include_compacted=False),
         wire=lambda r: {"messages": list(r)}, seed=_seed_messages, store_seed=_store_messages),
    Case("compression_boundary", "compression.boundary", {"session_key": "k", "source": "cli"},
         _w("compression_boundary", peer=ConversationPeer(session_key="k", source="cli")), session_key="k"),
    Case("compact_transcript", "compression.archive", COMPACTION.payload(), _w("compact_transcript", compaction=COMPACTION),
         wire=lambda r: {"value": r.value, "annotations": [a.as_wire() for a in r.annotations]}, seed=_seed_messages, store_seed=_store_messages),
    Case("publish_compression_child", "compression.publish", PUBLICATION.payload(),
         _w("publish_compression_child", publication=PUBLICATION),
         wire=lambda r: {"value": None, "worker_assignment": {"parent": r.parent, "session_id": r.session_id},
                         "annotations": [a.as_wire() for a in r.annotations]}, seed=_seed_compression_lock, store_seed=_store_compression_lock),
    Case("append_compression_transcript", "compression.append", APPEND, _w("append_compression_transcript", **APPEND),
         wire=_annotated, seed=_seed_messages, store_seed=_store_messages),
]


def _session(case: Case) -> dict[str, Any]:
    return {"session_key": case.session_key} if case.session_key else {}


def _seed(path: Any, case: Case) -> int:
    return seed_worker(path, extra=case.seed, **_session(case))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_verb_matches_m1(backend, scope_a, frozen, case):
    path = store_path(scope_a)
    epoch = _seed(path, case)
    store = backend.open(scope_a)
    idn = ident(epoch, case.sequence)
    legacy = (legacy_ledger if case.ledger else legacy_persist)(case.operation, case.payload, idn)
    before, _after = parity(path, legacy, lambda: case.call(store, idn), case.wire)
    assert before[0] == "ok", before


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_verb_is_one_transaction(backend, scope_a, frozen, case):
    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store, extra=case.store_seed, **_session(case))
    idn = ident(epoch, case.sequence)
    rolls_back(store, path, lambda: case.call(store, idn), *case.fail_at)
    one_transaction(store, lambda: case.call(store, idn))
