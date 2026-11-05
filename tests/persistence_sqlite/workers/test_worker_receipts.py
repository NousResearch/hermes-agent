"""The worker receipt transaction: the revision bump only when rows changed, and the retired operations, which no handler
takes and no verb sends."""

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
from persistence.workers.executions import WorkerTurnLease
from tests.persistence_sqlite.authority._support import dump
from tests.persistence_sqlite.workers._support import (
    EXECUTION, SESSION, frozen, ident, revision, seed_store, store_path,
)

__all__ = ["frozen"]


# ── a compression publication ────────────────────────────────────────────────

CHILD = "sess-worker-child"
LOCK = "pid=1:compress"
PUBLICATION = WorkerCompressionPublication(child_session_id=SessionId(CHILD), source="cli",
                                           messages=[{"role": "user", "content": "carried"}], compression_lock_holder=LOCK)


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


# ── retired operations ───────────────────────────────────────────────────────

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
