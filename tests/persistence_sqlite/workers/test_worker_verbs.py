"""The receipted worker verbs (session rows, transcript, titles, usage, compression, delegation ledger): each verb is one
transaction that a failure injected after its first statement leaves without a trace."""

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
    NOW, SESSION, frozen, ident, one_transaction, owner_started_at, rolls_back, seed_store, store_path, worker_birth,
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


PUBLICATION = WorkerCompressionPublication(child_session_id=SessionId(CHILD), source="cli",
                                           messages=[{"role": "user", "content": "carried"}], compression_lock_holder=LOCK)
COMPACTION = WorkerCompaction(messages=[{"role": "user", "content": "summary"}])
APPEND = {"messages": [{"role": "user", "content": "x"}], "compression_lock_holder": None, "turn_lease_holder": None,
          "turn_lease_ttl_seconds": 300.0}


@dataclass(frozen=True)
class Case:
    """One verb call: the verb, the sequence the call takes, the table and event a failure is injected at, the seeded
    session's key, and the seed through the backend."""

    name: str
    call: Callable[[Any, dict[str, Any]], Any]
    sequence: int = 1
    fail_at: tuple[str, str] = ("worker_receipts", "INSERT")
    session_key: str | None = None
    store_seed: Callable[[Any, int], None] | None = None


def _w(verb: str, **kwargs: Any) -> Callable[[Any, dict[str, Any]], Any]:
    return lambda store, idn: getattr(store.worker_executions, verb)(**idn, **kwargs)


def _ledger(verb: str, **kwargs: Any) -> Callable[[Any, dict[str, Any]], Any]:
    """A delegation ledger verb with this process as the verified worker."""
    return lambda store, idn: getattr(store.worker_executions, verb)(**idn, **kwargs, worker_pid=os.getpid(),
                                                                      worker_birth=worker_birth())


CASES = [
    Case("finish", _w("finish")),
    Case("set_session_title_source", _w("set_session_title_source", source="user"), store_seed=_store_title),
    Case("next_session_title", _w("next_session_title", base_title="Title"), store_seed=_store_title),
    Case("touch_session_activity",
         _w("touch_session_activity", activity=WorkerActivity(ts=NOW - 5, description="reading", provenance="tool"))),
    Case("clear_session_activity", _w("clear_session_activity")),
    Case("set_session_billing_route",
         _w("set_session_billing_route", route=WorkerBillingRoute(provider="p", base_url="u", billing_mode=None))),
    Case("latest_conversation_role", _w("latest_conversation_role"), store_seed=_store_messages),
    Case("message_role", _w("message_role", row_id=1), store_seed=_store_messages),
    Case("renew_compression_lock", _w("renew_compression_lock", lease=WorkerTurnLease(holder="pid=1:lock", ttl_seconds=30.0)),
         store_seed=_store_lock),
    Case("compression_lock_holder", _w("compression_lock_holder"), store_seed=_store_lock),
    Case("record_compression_cooldown",
         _w("record_compression_cooldown", cooldown=WorkerCooldown(cooldown_until=NOW + 60, error="boom"))),
    Case("restore_compression_cooldown",
         _w("restore_compression_cooldown", snapshot=WorkerCooldownSnapshot(session_exists=True, cooldown_until=NOW + 5, error="e"))),
    Case("clear_compression_cooldown", _w("clear_compression_cooldown")),
    Case("set_compression_fallback_streak", _w("set_compression_fallback_streak", streak=3)),
    Case("set_compression_ineffective_count", _w("set_compression_ineffective_count", count=2)),
    Case("set_compression_recovery_deadline", _w("set_compression_recovery_deadline", deadline=-5.0)),
    Case("compression_watermark", _w("compression_watermark"), store_seed=_store_messages),
    Case("dispatch_delegation",
         lambda store, idn: store.worker_executions.dispatch_delegation(
             **idn, dispatch=WorkerDelegationDispatch(**_dispatch_payload()), worker_pid=os.getpid(),
             worker_birth=worker_birth(), owner_started_at=owner_started_at())),
    Case("complete_delegation",
         _ledger("complete_delegation", completion=WorkerDelegationCompletion(event=_event(), result={"summary": "s"})),
         store_seed=_store_dispatch, sequence=2),
    Case("record_delegation_child",
         _ledger("record_delegation_child", child=WorkerDelegationChild(delegation_id=DELEGATION, entry={"task_index": 0, "status": "ok"})),
         store_seed=_store_dispatch, sequence=2),
    Case("read_delegation", _ledger("read_delegation", delegation_id=DELEGATION), store_seed=_store_completed, sequence=3),
    Case("claim_delegation_delivery",
         _ledger("claim_delegation_delivery", claim=WorkerDelegationClaim(delegation_id=DELEGATION, claim_id="c1")),
         store_seed=_store_completed, sequence=3),
    *(Case(f"{action}_delegation_delivery",
           _ledger(f"{action}_delegation_delivery", claim=WorkerDelegationClaim(delegation_id=DELEGATION, claim_id="c1")),
           store_seed=_store_claimed, sequence=4)
      for action in ("ack", "drop", "defer", "release")),
    Case("withdraw_delegation", _ledger("withdraw_delegation", delegation_id=DELEGATION), store_seed=_store_dispatch, sequence=2),
    Case("set_session_title", _w("set_session_title", title="New title", source="user")),
    Case("set_latest_user_api_content", _w("set_latest_user_api_content", content="hello", api_content="api"),
         store_seed=_store_messages),
    Case("set_message_api_content", _w("set_message_api_content", row_id=1, content="hello", api_content="api"),
         store_seed=_store_messages),
    Case("set_user_message_content", _w("set_user_message_content", row_id=1, content="edited"), store_seed=_store_messages),
    Case("create_session", _w("create_session", source="cli", model="m")),
    Case("end_session", _w("end_session", end_reason="user_exit")),
    Case("session_lifecycle", _w("session_lifecycle"), store_seed=_store_messages),
    Case("set_session_system_prompt", _w("set_session_system_prompt", system_prompt="sys")),
    Case("patch_session_sidecars", _w("patch_session_sidecars", patch={"_usage_anchor": 5})),
    Case("set_session_tool_pin", _w("set_session_tool_pin", tool_pin={"version": "1", "tools": ["a"]})),
    Case("append_transcript", _w("append_transcript", messages=[{"role": "user", "content": "x", "timestamp": NOW - 1}])),
    Case("record_usage", _w("record_usage", input_tokens=5, output_tokens=2, model="m", source="gui")),
    Case("record_auxiliary_usage", _w("record_auxiliary_usage", task="title", input_tokens=3)),
    Case("reopen_orphaned_compression", _w("reopen_orphaned_compression"), store_seed=_store_compressed),
    Case("compression_session_context", _w("compression_session_context", target_session_id=SESSION)),
    Case("compression_lineage_view", _w("compression_lineage_view", target_session_id=SESSION)),
    Case("compression_history",
         _w("compression_history", target_session_id=SESSION, include_ancestors=True, include_inactive=False,
            repair_alternation=True, include_row_ids=True, include_compacted=False),
         store_seed=_store_messages),
    Case("compression_boundary", _w("compression_boundary", peer=ConversationPeer(session_key="k", source="cli")), session_key="k"),
    Case("compact_transcript", _w("compact_transcript", compaction=COMPACTION), store_seed=_store_messages),
    Case("publish_compression_child", _w("publish_compression_child", publication=PUBLICATION),
         store_seed=_store_compression_lock),
    Case("append_compression_transcript", _w("append_compression_transcript", **APPEND), store_seed=_store_messages),
]


def _session(case: Case) -> dict[str, Any]:
    return {"session_key": case.session_key} if case.session_key else {}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_verb_is_one_transaction(backend, scope_a, frozen, case):
    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store, extra=case.store_seed, **_session(case))
    idn = ident(epoch, case.sequence)
    rolls_back(store, path, lambda: case.call(store, idn), *case.fail_at)
    one_transaction(store, lambda: case.call(store, idn))
