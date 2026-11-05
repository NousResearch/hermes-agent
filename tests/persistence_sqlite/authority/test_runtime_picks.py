"""``SessionRuntimeRepository`` on SQLite against main: the epoch, the mutation-receipt check and the deferred and chat
picks.

Each verb runs main's function and the backend verb on copies of one seed (same outcome, same rows), opens exactly one
transaction, and a write that fails right after its first mutating statement leaves every table unchanged. Main's
gateway callers that run their own SQL (the deferred pick, the dashboard's epoch read) are the oracle for the verbs
that absorbed it.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from persistence.authority.runtime import (
    ChatPickParams,
    DeferredPick,
    SessionRuntimeState,
    chat_pick_key,
    mutation_receipt_key,
)
from tests.persistence_sqlite.authority._admission_support import BackendSeed, begin_epoch
from tests.persistence_sqlite.authority._support import (
    assert_same,
    counted_transactions,
    dump,
    outcome,
    run_both,
    seed_legacy,
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


def assert_one_read(store: Any, call: Callable[[], Any]) -> None:
    """``call`` opens exactly one read context and no write."""
    with counted_transactions(store.holder()) as counts:
        outcome(call)
    assert counts == {"write": 0, "read": 1}


def build_seed(store: Any, seed: Callable[[Any], Any], *, legacy: bool) -> None:
    """Run ``seed(db)`` for ``store``: on the legacy store with ``legacy``, else with a :class:`BackendSeed`."""
    if legacy:
        seed_legacy(store.store_path, seed)
    else:
        seed(BackendSeed(store))


def _seed(store: Any, *, epoch: bool = True, receipt: bool = False, legacy: bool = True) -> None:
    def seed(db: Any) -> None:
        db.create_session(SID, source="telegram")
        if epoch:
            begin_epoch(db, OWNER)
        if receipt:
            from hermes_state_mutation_chat_pick import commit_chat_pick

            params = _chat_params()
            key, digest = chat_pick_key(PRINCIPAL, SID, params)
            commit_chat_pick(db, epoch=1, key=key, digest=digest, session_id=SID, params=_legacy(params),
                             result={"session_id": SID, "operation": "model", "model": "m-seeded"})
    build_seed(store, seed, legacy=legacy)


def _chat_params(request_id: str = "req-chat", *, model: str = "m-2", revision: int = 0, generation: int = 0) -> ChatPickParams:
    return ChatPickParams(request_id=request_id, operation="model", payload={"model": model},
                          expected_revision=revision, expected_generation=generation)


def _legacy(params: ChatPickParams) -> dict[str, Any]:
    return {"request_id": params.request_id, "operation": params.operation, "payload": dict(params.payload),
            "expected_revision": params.expected_revision, "expected_generation": params.expected_generation}


# ── begin_epoch / current_epoch ───────────────────────────────────────────────


@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("instance", [OWNER, ""])
def test_begin_epoch_matches_m1(store, legacy_store, seeded, instance):
    from hermes_state_runtime import begin_runtime_epoch

    _seed(store, epoch=seeded)
    legacy, ours = run_both(store.store_path, lambda db: begin_runtime_epoch(db, instance_id=instance),
                            lambda: store.runtime.begin_epoch(instance_id=instance))
    assert_same(legacy, ours)


def test_begin_epoch_is_one_write(store):
    _seed(store, legacy=False)
    assert_one_write(store, lambda: store.runtime.begin_epoch(instance_id="owner-b"))


@pytest.mark.parametrize("seeded", [False, True])
def test_current_epoch_matches_the_dashboard_read(store, legacy_store, seeded):
    _seed(store, epoch=seeded)

    def legacy(db: Any) -> Any:
        # main's epoch read in hermes_cli/web_server_sessions.py
        with db._read_ctx() as conn:
            row = conn.execute('SELECT epoch FROM runtime_epoch WHERE singleton=1').fetchone()
        return row[0] if row else None

    legacy_out, ours = run_both(store.store_path, legacy, store.runtime.current_epoch)
    assert_same(legacy_out, ours)
    assert ours[1] == (1 if seeded else None)
    assert_one_read(store, store.runtime.current_epoch)


# ── has_mutation_receipt ──────────────────────────────────────────────────────


@pytest.mark.parametrize("identity", [(PRINCIPAL, SID, "req-chat"), (PRINCIPAL, SID, "req-other"), ("user-2", SID, "req-chat"), ("", SID, "req-chat")])
def test_has_mutation_receipt_matches_m1(store, legacy_store, identity):
    from hermes_state_mutation_retirement import has_mutation_receipt

    _seed(store, receipt=True)
    legacy, ours = run_both(store.store_path, lambda db: has_mutation_receipt(db, *identity),
                            lambda: store.runtime.has_mutation_receipt(*identity))
    assert_same(legacy, ours)
    assert_one_read(store, lambda: store.runtime.has_mutation_receipt(PRINCIPAL, SID, "req-chat"))


# ── chat picks ────────────────────────────────────────────────────────────────


_CHAT_CASES = {
    "fresh": dict(params=_chat_params()),
    "exact_retry": dict(params=_chat_params(revision=0), receipt=True),
    "reused_request": dict(params=_chat_params(model="m-other"), receipt=True),
    "stale_epoch": dict(params=_chat_params(), epoch=2),
    "missing_session": dict(params=_chat_params(), session="s-none"),
    "revision_conflict": dict(params=_chat_params(revision=3)),
    "stale_generation": dict(params=_chat_params(generation=2)),
}


def _project_prepared(value: Any) -> Any:
    if isinstance(value, SessionRuntimeState):
        return {"row": (value.revision, value.generation)}
    return {"receipt": value}


@pytest.mark.parametrize("case", sorted(_CHAT_CASES))
def test_prepare_chat_pick_matches_m1(store, legacy_store, case):
    from hermes_state_mutation_chat_pick import prepare_chat_pick

    spec = _CHAT_CASES[case]
    _seed(store, receipt=spec.get("receipt", False))
    params, session, epoch = spec["params"], spec.get("session", SID), spec.get("epoch", 1)
    key, digest = chat_pick_key(PRINCIPAL, session, params)

    def legacy(db: Any) -> Any:
        found = prepare_chat_pick(db, epoch=epoch, key=key, digest=digest, session_id=session, params=_legacy(params))
        if "row" in found:
            return {"row": (found["row"]["runtime_revision"], found["row"]["runtime_generation"])}
        return found

    legacy_out, ours = run_both(store.store_path, legacy, lambda: store.runtime.prepare_chat_pick(
        epoch, key=key, digest=digest, session_id=session, params=params))
    assert_same(legacy_out, ours, _project_prepared)
    assert_one_read(store, lambda: store.runtime.prepare_chat_pick(epoch, key=key, digest=digest, session_id=session, params=params))


@pytest.mark.parametrize("case", sorted(_CHAT_CASES))
def test_commit_chat_pick_matches_m1(store, legacy_store, case):
    from hermes_state_mutation_chat_pick import commit_chat_pick

    spec = _CHAT_CASES[case]
    _seed(store, receipt=spec.get("receipt", False))
    params, session, epoch = spec["params"], spec.get("session", SID), spec.get("epoch", 1)
    key, digest = chat_pick_key(PRINCIPAL, session, params)
    result = {"session_id": session, "operation": "model", "model": params.payload["model"]}
    legacy, ours = run_both(
        store.store_path,
        lambda db: commit_chat_pick(db, epoch=epoch, key=key, digest=digest, session_id=session, params=_legacy(params), result=result),
        lambda: store.runtime.commit_chat_pick(epoch, key=key, digest=digest, session_id=session, params=params, result=result))
    assert_same(legacy, ours)


def test_commit_chat_pick_is_one_write(store):
    _seed(store, legacy=False)
    params = _chat_params()
    key, digest = chat_pick_key(PRINCIPAL, SID, params)
    assert_one_write(store, lambda: store.runtime.commit_chat_pick(
        1, key=key, digest=digest, session_id=SID, params=params, result={"session_id": SID, "model": "m-2"}))


# ── deferred picks ────────────────────────────────────────────────────────────


def _defer_args(operation: str, payload: dict[str, Any], request_id: str) -> tuple[str, str]:
    return mutation_receipt_key(PRINCIPAL, SID, request_id, operation=operation, payload=payload,
                                expected_revision=0, expected_generation=0)


_DEFER_CASES = {
    "model": [("model", {"model": "m-2"}, "r1")],
    "reasoning": [("reasoning", {"effort": "high"}, "r1")],
    "model_keeps_older_level": [("reasoning", {"effort": "low"}, "r1"), ("model", {"model": "m-2"}, "r2")],
    "model_with_own_level": [("reasoning", {"effort": "low"}, "r1"), ("model", {"model": "m-2", "reasoning": "high"}, "r2")],
    "level_drops_model_level": [("model", {"model": "m-2", "reasoning": "high"}, "r1"), ("reasoning", {"effort": "low"}, "r2")],
    "exact_retry": [("model", {"model": "m-2"}, "r1"), ("model", {"model": "m-2"}, "r1")],
    "reused_request": [("model", {"model": "m-2"}, "r1"), ("model", {"model": "m-3"}, "r1")],
    "invalid_payload": [("model", {"model": ""}, "r1")],
}


@pytest.mark.parametrize("case", sorted(_DEFER_CASES))
def test_defer_model_pick_matches_m1(store, legacy_store, case):
    from tests.persistence_sqlite._main_callers.session_deferred_pick import defer_pick

    *earlier, (operation, payload, request_id) = _DEFER_CASES[case]

    def seed(db: Any) -> None:
        authority = SimpleNamespace(db=db, epoch=1)
        for op, body, rid in earlier:
            defer_pick(authority, SimpleNamespace(subject=PRINCIPAL), SimpleNamespace(session_id=SID),
                       {"operation": op, "payload": body, "request_id": rid, "expected_revision": 0, "expected_generation": 0})

    _seed(store)
    seed_legacy(store.store_path, seed)
    key, digest = _defer_args(operation, payload, request_id)
    legacy, ours = run_both(
        store.store_path,
        lambda db: defer_pick(SimpleNamespace(db=db, epoch=1), SimpleNamespace(subject=PRINCIPAL), SimpleNamespace(session_id=SID),
                              {"operation": operation, "payload": payload, "request_id": request_id,
                               "expected_revision": 0, "expected_generation": 0}),
        lambda: store.runtime.defer_model_pick(1, key=key, digest=digest, session_id=SID, pick=DeferredPick(
            operation=operation, principal_id=PRINCIPAL, request_id=request_id, payload=payload)))
    assert_same(legacy, ours)


@pytest.mark.parametrize("epoch, session", [(2, SID), (1, "s-none")])
def test_defer_model_pick_fences_match_m1(store, legacy_store, epoch, session):
    from tests.persistence_sqlite._main_callers.session_deferred_pick import defer_pick

    _seed(store)
    payload = {"model": "m-2"}
    key, digest = mutation_receipt_key(PRINCIPAL, session, "r1", operation="model", payload=payload,
                                       expected_revision=0, expected_generation=0)
    legacy, ours = run_both(
        store.store_path,
        lambda db: defer_pick(SimpleNamespace(db=db, epoch=epoch), SimpleNamespace(subject=PRINCIPAL),
                              SimpleNamespace(session_id=session),
                              {"operation": "model", "payload": payload, "request_id": "r1",
                               "expected_revision": 0, "expected_generation": 0}),
        lambda: store.runtime.defer_model_pick(epoch, key=key, digest=digest, session_id=session, pick=DeferredPick(
            operation="model", principal_id=PRINCIPAL, request_id="r1", payload=payload)))
    assert_same(legacy, ours)


def test_defer_model_pick_is_one_write(store):
    _seed(store, legacy=False)
    key, digest = _defer_args("model", {"model": "m-2"}, "r1")
    assert_one_write(store, lambda: store.runtime.defer_model_pick(1, key=key, digest=digest, session_id=SID, pick=DeferredPick(
        operation="model", principal_id=PRINCIPAL, request_id="r1", payload={"model": "m-2"})))


def _seed_deferred(store: Any, *, legacy: bool = True) -> str:
    """Seed a deferred model pick (the gateway caller on the legacy store, else the verb) and return the stored slot
    value."""
    _seed(store, legacy=legacy)
    stored: list[str] = []

    def seed(db: Any) -> None:
        if isinstance(db, BackendSeed):
            key, digest = _defer_args("model", {"model": "m-2"}, "r1")
            db.store.runtime.defer_model_pick(1, key=key, digest=digest, session_id=SID, pick=DeferredPick(
                operation="model", principal_id=PRINCIPAL, request_id="r1", payload={"model": "m-2"}))
            stored.append(db.store.runtime.deferred_pick(SID).stored)
            return
        from tests.persistence_sqlite._main_callers.session_deferred_pick import defer_pick
        from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX

        defer_pick(SimpleNamespace(db=db, epoch=1), SimpleNamespace(subject=PRINCIPAL), SimpleNamespace(session_id=SID),
                   {"operation": "model", "payload": {"model": "m-2"}, "request_id": "r1",
                    "expected_revision": 0, "expected_generation": 0})
        stored.append(db.get_meta(DEFERRED_PICK_PREFIX + SID))

    build_seed(store, seed, legacy=legacy)
    return stored[0]


@pytest.mark.parametrize("present", [True, False])
def test_deferred_pick_reads_the_stored_slot(store, legacy_store, present):
    from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX

    stored = _seed_deferred(store) if present else _seed(store)
    legacy, ours = run_both(store.store_path, lambda db: db.get_meta(DEFERRED_PICK_PREFIX + SID),
                            lambda: store.runtime.deferred_pick(SID))
    assert_same(legacy, ours, lambda found: None if found is None else found.stored)
    if present:
        assert ours[1].stored == stored and ours[1].pick["payload"] == {"model": "m-2"}
    assert_one_read(store, lambda: store.runtime.deferred_pick(SID))


@pytest.mark.parametrize("current", [True, False])
def test_forget_deferred_pick_matches_m1(store, legacy_store, current):
    from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX

    stored = _seed_deferred(store)
    compare = stored if current else stored + " "
    # main's deferred pick in gateway/session_deferred_pick.py
    legacy, ours = run_both(
        store.store_path,
        lambda db: db._execute_write(lambda conn: conn.execute(
            'DELETE FROM state_meta WHERE key=? AND value=?', (DEFERRED_PICK_PREFIX + SID, compare))) and None,
        lambda: store.runtime.forget_deferred_pick(SID, stored=compare))
    assert_same(legacy, ours)


def test_forget_deferred_pick_is_one_write(store):
    stored = _seed_deferred(store, legacy=False)
    assert_one_write(store, lambda: store.runtime.forget_deferred_pick(SID, stored=stored))


@pytest.mark.parametrize("current", [True, False])
def test_mark_deferred_pick_asked_matches_m1(store, legacy_store, current):
    import json

    from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX
    from hermes_state_runtime import _json

    stored = _seed_deferred(store)
    compare = stored if current else stored + " "
    # main's deferred pick in gateway/session_deferred_pick.py (the pick as stored, now asked)
    legacy, ours = run_both(
        store.store_path,
        lambda db: db._execute_write(lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=? AND value=?',
            (_json({**json.loads(stored), 'awaiting_confirm': 'tok-1'}), DEFERRED_PICK_PREFIX + SID, compare))) and None,
        lambda: store.runtime.mark_deferred_pick_asked(SID, stored=compare, confirm="tok-1"))
    assert_same(legacy, ours)


def test_mark_deferred_pick_asked_refuses_an_undecodable_compare_value(store):
    from persistence import AuthorityRefusal

    _seed(store, legacy=False)
    with pytest.raises(AuthorityRefusal) as caught:
        store.runtime.mark_deferred_pick_asked(SID, stored="not json", confirm="tok-1")
    assert caught.value.reason == "invalid_params"


def test_mark_deferred_pick_asked_is_one_write(store):
    stored = _seed_deferred(store, legacy=False)
    assert_one_write(store, lambda: store.runtime.mark_deferred_pick_asked(SID, stored=stored, confirm="tok-1"))
