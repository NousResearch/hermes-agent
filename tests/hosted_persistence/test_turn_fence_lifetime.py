"""A claimed turn's fence covers all the work the turn started, for that work's whole lifetime.

The turn's identity is one value: the bound ``TurnScope`` (session, turn, fence), plus the store that issued the claim
(``agent.runtime_scope.bind_turn(..., store=)``). Work the turn starts (a copied asyncio task, a thread, a queued token
update) carries that identity with it, so once another owner takes the profile over or a newer claim moves the
session's generation, that work's writes are refused, also after the turn's body returned. The owner's own work
(claims, settlement, the drains it schedules) runs as an owner operation under the admission ledger's own checks, and
work designed to outlive the turn (async delegation, the title upgrade, the background review, the compaction pass)
keeps only the owner-epoch half of the fence.
"""

from __future__ import annotations

import asyncio
import contextvars
import sqlite3
import threading
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace

import pytest

from agent.runtime_scope import bind_turn, claim_fence, outlive_turn, owner_operation
from hermes_scope import Fence, current_turn
from persistence import FenceStale, SessionId, TurnFenceSuperseded
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import TokenCounts, UsageDelta
from persistence.core.fences import writing_turn
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder

S = SessionId("fence-life")


def _elsewhere(fn, *args, **kwargs):
    """Run ``fn`` as another writer would: in a fresh context, outside any turn."""
    return contextvars.Context().run(lambda: fn(*args, **kwargs))


@pytest.fixture
def claimed(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path), eager=True)
    try:
        store.sessions.create(NewSession(session_id=S, source="cli"))
        epoch = store.runtime.begin_epoch(instance_id="one")
        store.admissions.admit(epoch=epoch, principal_id="u", session_id=S, request_id="r1", payload={})
        row = store.admissions.claim(epoch=epoch, session_id=S)
        yield store, epoch, row
    finally:
        store.close()


def _turn(store, row):
    """The turn bound as a single-profile gateway binds it: no scope, the claim's store named."""
    return bind_turn(S, row.admission_id, claim_fence(row), store=store.scope)


def _take_over(store) -> None:
    _elsewhere(store.runtime.begin_epoch, instance_id="two")


def _next_claim(store, epoch, row):
    """The same owner settles ``row`` and claims the session's next input: a new generation."""
    def run():
        store.admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=row.generation, outcome="completed")
        store.admissions.admit(epoch=epoch, principal_id="u", session_id=S, request_id="r2", payload={})
        following = store.admissions.claim(epoch=epoch, session_id=S)
        assert following.generation > row.generation
        return following
    return _elsewhere(run)


def _append(store, text: str) -> None:
    store.messages.append(S, NewMessage(role="assistant", content=text))


def _contents(store) -> list:
    return [m.content for m in store.messages.read_transcript(S)]


def _tokens(n: int) -> UsageDelta:
    return UsageDelta(tokens=TokenCounts(input_tokens=n), api_call_count=1)


# --- work the turn started keeps the turn's fence after the body exits ---------------------------

def test_a_copied_task_cannot_land_a_late_reply_after_takeover(claimed):
    store, _epoch, row = claimed
    outcome = []

    async def scenario():
        go = asyncio.Event()

        async def late():
            await go.wait()
            assert current_turn() is not None  # still the turn's work
            try:
                _append(store, "stale late reply")
                outcome.append("written")
            except TurnFenceSuperseded:
                outcome.append("refused")
        with _turn(store, row):
            task = asyncio.create_task(late())
        _take_over(store)
        go.set()
        await task
    asyncio.run(scenario())
    assert outcome == ["refused"]
    assert _contents(store) == []


def test_a_copied_task_of_a_turn_still_holding_its_lease_writes(claimed):
    store, _epoch, row = claimed

    async def scenario():
        go = asyncio.Event()

        async def late():
            await go.wait()
            _append(store, "late but current")
        with _turn(store, row):
            task = asyncio.create_task(late())
        go.set()
        await task
    asyncio.run(scenario())
    assert _contents(store) == ["late but current"]


def test_a_thread_the_turn_started_stays_fenced_after_the_turn_returns(claimed):
    store, epoch, row = claimed
    results = []

    def background():
        try:
            _append(store, "late")
            results.append("written")
        except TurnFenceSuperseded:
            results.append("refused")
    with _turn(store, row):
        first, second = contextvars.copy_context(), contextvars.copy_context()
    # The turn's body returned; the turn still holds the lease, so its thread writes.
    thread = threading.Thread(target=first.run, args=(background,))
    thread.start()
    thread.join(10)
    _next_claim(store, epoch, row)
    thread = threading.Thread(target=second.run, args=(background,))
    thread.start()
    thread.join(10)
    assert results == ["written", "refused"]
    assert _contents(store) == ["late"]


def test_writes_outside_a_claimed_turn_or_to_another_store_are_unaffected(claimed, tmp_path):
    store, epoch, row = claimed
    _next_claim(store, epoch, row)
    _append(store, "outside")  # no turn bound: maintenance, notes, CLI
    with bind_turn(S, row.admission_id, claim_fence(row)):
        _append(store, "no store named")  # no scope and no store: nothing to fence
    with _turn(store, row), owner_operation():
        _append(store, "owner")  # the owner's own work in a turn's context
    other_home = tmp_path / "other"
    other_home.mkdir()
    other = open_persistence(profile_scope(home=other_home), eager=True)
    try:
        other.sessions.create(NewSession(session_id=S, source="cli"))
        with _turn(store, row):
            other.messages.append(S, NewMessage(role="user", content="another store"))
            with pytest.raises(TurnFenceSuperseded):
                _append(store, "issuing store")
        assert [m.content for m in other.messages.read_transcript(S)] == ["another store"]
    finally:
        other.close()
    assert _contents(store) == ["outside", "no store named", "owner"]


def test_a_transient_io_error_in_the_fence_check_is_retried_like_begin(claimed, monkeypatch):
    """The check runs before the callback, so its reads keep the pre-callback disk I/O retry."""
    import persistence.sqlite._holder as holder_module  # the SQLite backend's write path: the seam under test
    store, _epoch, row = claimed
    real, calls = holder_module.check_turn_fence, []

    def flaky(conn, fence):
        calls.append(fence)
        if fence is not None and len([c for c in calls if c is not None]) == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return real(conn, fence)
    monkeypatch.setattr(holder_module, "check_turn_fence", flaky)
    with _turn(store, row):
        _append(store, "kept")
    assert len([c for c in calls if c is not None]) == 2
    assert _contents(store) == ["kept"]


# --- queued token accounting carries the claim it was queued under ------------------------------

def test_a_superseded_turns_queued_tokens_are_refused_like_its_synchronous_ones(claimed):
    store, _epoch, row = claimed
    with _turn(store, row):
        _take_over(store)
        store.usage.queue_usage(S, _tokens(17))
        assert store.usage.flush_usage()
    assert store.usage.flush_usage()
    assert store.sessions.get(S).input_tokens == 0


def test_queued_tokens_of_a_live_turn_land_after_its_body_returned(claimed):
    store, _epoch, row = claimed
    holder = raw_holder(store)
    gate = threading.Event()
    original = holder._apply_token_batch

    def held(batch):
        assert gate.wait(10)
        return original(batch)
    holder._apply_token_batch = held
    try:
        with _turn(store, row):
            store.usage.queue_usage(S, _tokens(5))
        gate.set()  # the writer applies after the body exited, in its own thread
        assert store.usage.flush_usage()
    finally:
        holder._apply_token_batch = original
    assert store.sessions.get(S).input_tokens == 5


def test_queued_tokens_never_merge_across_claims(claimed):
    """A delta queued by the superseded claim and one queued by its successor, in one writer batch: the writer refuses
    the first and lands the second, never a merged sum under either fence."""
    store, epoch, row = claimed
    holder = raw_holder(store)
    with _turn(store, row):
        old = writing_turn()
    following = _next_claim(store, epoch, row)
    with _turn(store, following):
        new = writing_turn()
    batch = [(S, dict(input_tokens=3, api_call_count=1), old), (S, dict(input_tokens=4, api_call_count=1), new)]
    assert holder._coalesce_token_deltas(list(batch)) == batch
    holder._apply_token_batch(batch)
    assert store.sessions.get(S).input_tokens == 4
    # Same claim, same route: still one merged write.
    assert len(holder._coalesce_token_deltas([batch[1], batch[1]])) == 1


def test_tokens_queued_outside_any_turn_are_unfenced(claimed):
    store, _epoch, _row = claimed
    _take_over(store)
    store.usage.queue_usage(S, _tokens(2))
    assert store.usage.flush_usage()
    assert store.sessions.get(S).input_tokens == 2


# --- the owner's own work runs as an owner operation, under its own checks -----------------------

def test_settlement_from_a_copied_turn_context_runs_as_the_owners_operation(claimed):
    store, epoch, row = claimed
    with _turn(store, row):
        copied = contextvars.copy_context()
    following = _next_claim(store, epoch, row)

    def settle():
        with owner_operation():
            assert current_turn() is None
            return store.admissions.settle(epoch=epoch, admission_id=following.admission_id,
                                           generation=following.generation, outcome="completed")
    assert copied.run(settle).status == "terminal"


def test_an_owner_operation_keeps_the_ledgers_own_fence(claimed):
    store, epoch, row = claimed
    _take_over(store)

    def settle():
        with owner_operation():
            return store.admissions.settle(epoch=epoch, admission_id=row.admission_id, generation=row.generation,
                                           outcome="completed")
    with pytest.raises(FenceStale):
        settle()


def test_a_drain_a_turn_schedules_runs_as_the_owners_operation(claimed, monkeypatch):
    """A follow-up a turn admitted is drained (claimed, settled) as the owner's work, not under that turn's fence."""
    import gateway.session_retirement as retirement
    import gateway.session_runtime_workers as workers
    from gateway.session_authority import SessionAuthority
    store, _epoch, row = claimed
    monkeypatch.setattr(retirement, "restore_retired", lambda *a: None)
    monkeypatch.setattr(workers, "persist_work_count", lambda *a: None)
    seen = []

    async def drain(ref):
        seen.append(current_turn())

    async def scenario():
        live = SimpleNamespace(rescan=False, task=None, background_wait=None)
        authority = SimpleNamespace(sessions={S: live}, _drain=drain)
        with _turn(store, row):
            assert current_turn() is not None
            SessionAuthority._schedule(authority, SimpleNamespace(session_id=S))
        await live.task
    asyncio.run(scenario())
    assert seen == [None]


def test_the_turn_identity_is_immutable(claimed):
    store, _epoch, row = claimed
    with _turn(store, row):
        mark = current_turn()
    with pytest.raises(AttributeError):
        mark.store = None
    assert (mark.session, mark.turn, mark.fence, mark.store) == (
        S, row.admission_id, Fence(row.owner_epoch, row.generation), store.scope.store_key)


# --- the gateway binds every claimed turn on the store that issued the claim ---------------------

def test_an_admitted_turn_is_fenced_on_its_authoritys_store(claimed, monkeypatch):
    import gateway.session_ingress as ingress
    store, _epoch, row = claimed

    @asynccontextmanager
    async def no_producer(*_a):
        yield

    async def no_scope(*_a):
        return nullcontext()
    monkeypatch.setattr(ingress, "producer_turn", no_producer)
    monkeypatch.setattr(ingress, "_execution_scope", no_scope)
    seen = []

    async def scenario():
        authority = SimpleNamespace(persistence=store)
        record = {"admission_id": row.admission_id, "owner_epoch": row.owner_epoch, "generation": row.generation}
        async with ingress._admission_scope(authority, SimpleNamespace(session_id=S), None, record, None, {}):
            seen.append(current_turn())
    asyncio.run(scenario())
    assert seen[0].store == store.scope.store_key and seen[0].turn == row.admission_id


# --- work designed to outlive the turn keeps the owner half of the fence -------------------------

def test_work_that_outlives_the_turn_survives_the_next_claim_but_not_a_takeover(claimed):
    store, epoch, row = claimed
    with _turn(store, row):
        copied = contextvars.copy_context()

    def report(text):
        with outlive_turn():
            _append(store, text)
    _next_claim(store, epoch, row)
    with pytest.raises(TurnFenceSuperseded):  # without outlive_turn the next claim fences it
        copied.run(_append, store, "plain late write")
    copied.run(report, "delegation result")
    _take_over(store)
    with pytest.raises(TurnFenceSuperseded):
        copied.run(report, "after takeover")
    assert _contents(store) == ["delegation result"]


def test_an_async_delegation_runs_as_work_that_outlives_its_turn(claimed, monkeypatch):
    import tools.async_delegation as ad
    store, _epoch, row = claimed
    seen, done = [], threading.Event()

    def runner():
        seen.append(current_turn())
        return {"status": "completed"}

    def finalize(*_a, **_k):
        seen.append(current_turn())
        done.set()
    monkeypatch.setattr(ad, "_persist_dispatch", lambda record: None)
    monkeypatch.setattr(ad, "_finalize", finalize)
    with _turn(store, row):
        handle = ad.dispatch_async_delegation(goal="g", context=None, toolsets=None, role="leaf", model=None,
                                              session_key="", parent_session_id=S, runner=runner)
    assert handle["status"] == "dispatched"
    assert done.wait(10)
    # The runner and the completion it reports back with alike.
    assert [mark.outlives for mark in seen] == [True, True] and seen[0].turn == row.admission_id


def test_a_stalled_delegations_forced_completion_outlives_its_turn(claimed, monkeypatch):
    import tools.async_delegation as ad
    store, _epoch, row = claimed
    seen = []
    monkeypatch.setattr(ad, "_finalize", lambda *a: seen.append(current_turn()))
    with _turn(store, row):
        contextvars.copy_context().run(ad._finalize_outliving_turn, "d1", {}, "stalled")
    assert seen[0].outlives and seen[0].turn == row.admission_id


def test_the_title_upgrade_runs_as_work_that_outlives_its_turn(claimed, monkeypatch):
    from agent import title_generator as tg
    store, _epoch, row = claimed
    seen = []
    monkeypatch.setattr(tg, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(tg, "_model_title_upgrade_enabled", lambda: True)
    monkeypatch.setattr(tg, "_kanban_task_title", lambda: None)
    monkeypatch.setattr(tg, "apply_instant_title", lambda *a, **k: None)
    monkeypatch.setattr(tg, "title_upgrade_must_wait_for_turn", lambda runtime: False)
    monkeypatch.setattr(tg, "auto_title_session", lambda *a, **k: seen.append(current_turn()))
    with _turn(store, row):
        thread = tg.maybe_auto_title(object(), S, "Explain transactional leases")
    thread.join(10)
    assert seen[0].outlives and seen[0].turn == row.admission_id


def test_a_background_review_runs_as_work_that_outlives_its_turn(claimed, monkeypatch):
    import types

    import agent.background_review as br
    import run_agent
    store, _epoch, row = claimed
    seen, done = [], threading.Event()

    def target():
        seen.append(current_turn())
        done.set()
    monkeypatch.setattr(br, "prepare_background_review_run", lambda agent: object())
    monkeypatch.setattr(br, "spawn_background_review_thread", lambda *a, **k: (target, ""))
    agent = types.SimpleNamespace(_maybe_requeue_preempted_review=lambda *a: None)
    spawn = types.MethodType(run_agent.AIAgent._spawn_background_review_now, agent)
    with _turn(store, row):
        spawn([])
    assert done.wait(10)
    assert seen[0].outlives and seen[0].turn == row.admission_id


def test_a_prepared_compaction_pass_runs_as_work_that_outlives_its_turn(claimed, monkeypatch):
    from agent.prepared_compaction import PreparedCompaction
    store, _epoch, row = claimed
    seen = []
    monkeypatch.setattr(PreparedCompaction, "_run_pass", lambda self, *a: seen.append(current_turn()))
    with _turn(store, row):
        PreparedCompaction._run_pass_outliving_turn(object.__new__(PreparedCompaction), None, [], None, None, "test")
    assert seen[0].outlives and seen[0].turn == row.admission_id
