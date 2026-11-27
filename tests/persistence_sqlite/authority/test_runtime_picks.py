"""``SessionRuntimeRepository`` on SQLite: the epoch and the chat and deferred pick writes each open exactly one write
transaction, which a write that fails right after its first mutating statement leaves unchanged, and marking a deferred
pick asked refuses an undecodable compare value.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import pytest

from persistence.authority.runtime import (
    ChatPickParams,
    DeferredPick,
    chat_pick_key,
    mutation_receipt_key,
)
from tests.persistence_sqlite.authority._admission_support import BackendSeed, begin_epoch
from tests.persistence_sqlite.authority._support import (
    counted_transactions,
    dump,
    outcome,
)

PRINCIPAL = "user-1"
SID = "s-pick"
OWNER = "owner-a"


class InjectedFailure(RuntimeError):
    """Raised inside a write right after its first mutating statement."""


class _FailAfterFirstWrite:
    """A writer connection that raises right after the first INSERT, UPDATE or DELETE it runs."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def _run(self, method: str, sql: str, *args: Any) -> Any:
        result = getattr(self._conn, method)(sql, *args)
        if sql.lstrip().split(None, 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            raise InjectedFailure(sql)
        return result

    def execute(self, sql: str, *args: Any) -> Any:
        return self._run("execute", sql, *args)

    def executemany(self, sql: str, *args: Any) -> Any:
        return self._run("executemany", sql, *args)


@contextmanager
def failing_after_first_write(holder: Any) -> Iterator[None]:
    """Every write transaction on ``holder`` fails right after its first mutating statement."""
    write = holder._execute_write

    def failing(fn: Callable[[Any], Any], *args: Any, **kwargs: Any) -> Any:
        return write(lambda conn: fn(_FailAfterFirstWrite(conn)), *args, **kwargs)

    holder._execute_write = failing
    try:
        yield
    finally:
        del holder._execute_write


def assert_one_write(store: Any, call: Callable[[], Any]) -> None:
    """``call`` is one write transaction: a failure after its first mutating statement leaves every table as it
    was, and the call itself opens exactly one write and no read."""
    holder = store.holder()
    before = dump(store.store_path)
    with failing_after_first_write(holder), pytest.raises(InjectedFailure):
        call()
    assert dump(store.store_path) == before
    with counted_transactions(holder) as counts:
        assert outcome(call)[0] == "ok"
    assert counts == {"write": 1, "read": 0}


def build_seed(store: Any, seed: Callable[[Any], Any]) -> None:
    """Run ``seed(db)`` for ``store`` with a :class:`BackendSeed`."""
    seed(BackendSeed(store))


def _seed(store: Any) -> None:
    def seed(db: Any) -> None:
        db.create_session(SID, source="telegram")
        begin_epoch(db, OWNER)
    build_seed(store, seed)


def _chat_params(request_id: str = "req-chat", *, model: str = "m-2", revision: int = 0, generation: int = 0) -> ChatPickParams:
    return ChatPickParams(request_id=request_id, operation="model", payload={"model": model},
                          expected_revision=revision, expected_generation=generation)


# ── begin_epoch ───────────────────────────────────────────────────────────────


def test_begin_epoch_is_one_write(store):
    _seed(store)
    assert_one_write(store, lambda: store.runtime.begin_epoch(instance_id="owner-b"))


# ── chat picks ────────────────────────────────────────────────────────────────


def test_commit_chat_pick_is_one_write(store):
    _seed(store)
    params = _chat_params()
    key, digest = chat_pick_key(PRINCIPAL, SID, params)
    assert_one_write(store, lambda: store.runtime.commit_chat_pick(
        1, key=key, digest=digest, session_id=SID, params=params, result={"session_id": SID, "model": "m-2"}))


# ── deferred picks ────────────────────────────────────────────────────────────


def _defer_args(operation: str, payload: dict[str, Any], request_id: str) -> tuple[str, str]:
    return mutation_receipt_key(PRINCIPAL, SID, request_id, operation=operation, payload=payload,
                                expected_revision=0, expected_generation=0)


def test_defer_model_pick_is_one_write(store):
    _seed(store)
    key, digest = _defer_args("model", {"model": "m-2"}, "r1")
    assert_one_write(store, lambda: store.runtime.defer_model_pick(1, key=key, digest=digest, session_id=SID, pick=DeferredPick(
        operation="model", principal_id=PRINCIPAL, request_id="r1", payload={"model": "m-2"})))


def _seed_deferred(store: Any) -> str:
    """Seed a deferred model pick through the verb and return the stored slot value."""
    _seed(store)
    key, digest = _defer_args("model", {"model": "m-2"}, "r1")
    store.runtime.defer_model_pick(1, key=key, digest=digest, session_id=SID, pick=DeferredPick(
        operation="model", principal_id=PRINCIPAL, request_id="r1", payload={"model": "m-2"}))
    return store.runtime.deferred_pick(SID).stored


def test_forget_deferred_pick_is_one_write(store):
    stored = _seed_deferred(store)
    assert_one_write(store, lambda: store.runtime.forget_deferred_pick(SID, stored=stored))


def test_mark_deferred_pick_asked_refuses_an_undecodable_compare_value(store):
    from persistence import AuthorityRefusal

    _seed(store)
    with pytest.raises(AuthorityRefusal) as caught:
        store.runtime.mark_deferred_pick_asked(SID, stored="not json", confirm="tok-1")
    assert caught.value.reason == "invalid_params"


def test_mark_deferred_pick_asked_is_one_write(store):
    stored = _seed_deferred(store)
    assert_one_write(store, lambda: store.runtime.mark_deferred_pick_asked(SID, stored=stored, confirm="tok-1"))
