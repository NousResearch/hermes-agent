"""Regressions for PendingCompletionRegistry (#70300).

Every test here maps to a defect found in review on PR #73469:

* lifetime capacity ceiling     -- teknium1, 2026-07-30
* retry indexes / backoff       -- teknium1, 2026-07-30
* duplicate vs capacity refusal -- single-value ambiguity the registry exists
                                   to remove
* terminal-state overwrite      -- supersede()/mark_unroutable() had no guard
* shutdown waiter resolution    -- signal_stop() left futures unresolved
* routing-key sentinel          -- asserted against the real key builder, not
                                   against hand-written tuples

Deliberately free of any Hypothesis dependency so it runs in CI unconditionally;
the property-based state machine lives in
``tests/gateway/test_pending_registry_properties.py``.
"""

from __future__ import annotations

import asyncio

from gateway.run import GatewayRunner

Registry = GatewayRunner.PendingCompletionRegistry
Disposition = GatewayRunner.CompletionDisposition
State = Registry.State
CAPACITY = Registry.BATCH_CAPACITY
MAX_ATTEMPTS = Registry.MAX_ATTEMPTS
batch_key = GatewayRunner._completion_notification_batch_key


def run_sync(coro):
    return asyncio.run(coro)


def _payload() -> dict:
    return {"synth_text": "text", "evt": {}}


# --------------------------------------------------------------- capacity


def test_completions_beyond_capacity_are_accepted_over_process_lifetime() -> None:
    """The cap bounds concurrency, not the lifetime of the gateway.

    Regression: ``BATCH_CAPACITY`` was compared against ``len(_entries)`` while
    terminal entries were never evicted, so completion number 51 and every one
    after it was refused for the remaining life of the process.
    """

    async def _test():
        reg = Registry()
        for i in range(CAPACITY * 4):
            ident = f"proc_{i:04d}"
            future = reg.enqueue(ident, ("r1",), _payload())
            assert future is not None, f"completion #{i} refused"
            reg.claim_batch([ident], f"b{i}")
            reg.deliver(ident)
            assert reg.snapshot()[ident] is State.DELIVERED

    run_sync(_test())


def test_capacity_bounds_live_entries_only() -> None:
    async def _test():
        reg = Registry()
        for i in range(CAPACITY):
            assert reg.enqueue(f"proc_{i:04d}", ("r1",), _payload()) is not None
        assert reg.live_count() == CAPACITY
        assert reg.enqueue("proc_overflow", ("r1",), _payload()) is None
        assert reg.rejection_reason("proc_overflow") == "capacity"

        # Retiring one live entry frees exactly one slot.
        reg.claim_batch(["proc_0000"], "b0")
        reg.deliver("proc_0000")
        assert reg.live_count() == CAPACITY - 1
        assert reg.enqueue("proc_overflow", ("r1",), _payload()) is not None

    run_sync(_test())


def test_terminal_entries_are_evicted_beyond_retention() -> None:
    """Terminal entries are a bounded dedupe memory, not an unbounded log."""

    async def _test():
        reg = Registry()
        reg.TERMINAL_RETENTION = 8
        try:
            for i in range(60):
                ident = f"proc_{i:04d}"
                reg.enqueue(ident, ("r1",), _payload())
                reg.claim_batch([ident], f"b{i}")
                reg.deliver(ident)
            assert len(reg.snapshot()) <= reg.TERMINAL_RETENTION + 1
        finally:
            del reg.TERMINAL_RETENTION

    run_sync(_test())


# -------------------------------------------------------------- refusals


def test_duplicate_enqueue_is_distinguishable_from_capacity() -> None:
    """Both refusals return ``None``; the reason must still be recoverable.

    Mapping both onto ``DROPPED_OVERFLOW`` reintroduced exactly the one-value
    ambiguity (``Optional[bool]``) that motivated this registry.
    """

    async def _test():
        reg = Registry()
        assert reg.enqueue("proc_a", ("r1",), _payload()) is not None
        assert reg.enqueue("proc_a", ("r1",), _payload()) is None
        assert reg.rejection_reason("proc_a") == "duplicate"

        # Once terminal, the identity is enqueueable again.
        reg.claim_batch(["proc_a"], "b1")
        reg.deliver("proc_a")
        assert reg.rejection_reason("proc_a") == "capacity"
        assert reg.enqueue("proc_a", ("r1",), _payload()) is not None

    run_sync(_test())


# ----------------------------------------------------------------- retry


def test_retry_backoff_grows_then_exhausts_into_failed() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        deadlines = []
        for attempt in range(MAX_ATTEMPTS):
            reg.claim_batch(["proc_a"], f"b{attempt}")
            deadline = reg.retry("proc_a")
            if deadline is None:
                break
            deadlines.append(deadline)
            assert reg.snapshot()["proc_a"] is State.PENDING

        assert reg.snapshot()["proc_a"] is State.FAILED
        assert len(deadlines) == MAX_ATTEMPTS - 1
        assert deadlines == sorted(deadlines), "backoff must be non-decreasing"

    run_sync(_test())


def test_retry_on_unclaimed_entry_is_noop() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        assert reg.retry("proc_a") is None
        assert reg.snapshot()["proc_a"] is State.PENDING

    run_sync(_test())


# ------------------------------------------------------- terminal guards


def test_supersede_never_overwrites_a_terminal_state() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        reg.claim_batch(["proc_a"], "b1")
        reg.deliver("proc_a")
        reg.supersede("proc_a")
        assert reg.snapshot()["proc_a"] is State.DELIVERED

    run_sync(_test())


def test_mark_unroutable_never_overwrites_a_terminal_state() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        reg.claim_batch(["proc_a"], "b1")
        reg.deliver("proc_a")
        reg.mark_unroutable("proc_a")
        assert reg.snapshot()["proc_a"] is State.DELIVERED

    run_sync(_test())


# -------------------------------------------------------------- shutdown


def test_signal_stop_resolves_every_pending_waiter() -> None:
    """STOPPING is terminal, so no waiter may be left parked on it.

    Regression: ``signal_stop()`` transitioned the entries but never resolved
    their futures, so a watcher cancelled inside the batch window awaited a
    Future nothing would complete.
    """

    async def _test():
        reg = Registry()
        pending = reg.enqueue("proc_pending", ("r1",), _payload())
        claimed = reg.enqueue("proc_claimed", ("r1",), _payload())
        done = reg.enqueue("proc_done", ("r1",), _payload())
        reg.claim_batch(["proc_claimed"], "b1")
        reg.claim_batch(["proc_done"], "b2")
        reg.deliver("proc_done")

        stopped = reg.signal_stop()
        stopped_ids = {entry["identity"] for entry in stopped}
        assert stopped_ids == {"proc_pending", "proc_claimed"}
        assert reg.snapshot()["proc_done"] is State.DELIVERED
        assert reg.is_stopping is True

        reg.resolve_futures(stopped, Disposition.SHUTTING_DOWN)
        assert await pending is Disposition.SHUTTING_DOWN
        assert await claimed is Disposition.SHUTTING_DOWN
        assert not done.done(), "terminal entry's waiter must not be touched"
        done.cancel()

    run_sync(_test())


def test_resolve_futures_is_idempotent() -> None:
    async def _test():
        reg = Registry()
        future = reg.enqueue("proc_a", ("r1",), _payload())
        entries = [reg._entries["proc_a"]]
        reg.resolve_futures(entries, Disposition.DELIVERED)
        reg.resolve_futures(entries, Disposition.RETRY)
        assert await future is Disposition.DELIVERED

    run_sync(_test())


# ------------------------------------------------------------ routing key


def test_none_and_empty_session_key_produce_distinct_routing_keys() -> None:
    """Asserted against the real key builder, not hand-written tuples."""
    none_key = batch_key({"session_key": None, "platform": "discord"})
    empty_key = batch_key({"session_key": "", "platform": "discord"})
    assert none_key != empty_key


def test_routing_key_separates_every_delivery_field() -> None:
    base = {
        "session_key": "s", "platform": "discord", "chat_type": "dm",
        "chat_id": "c1", "thread_id": "t1", "user_id": "u1",
    }
    for field in base:
        other = dict(base)
        other[field] = "different"
        assert batch_key(base) != batch_key(other), f"{field} not in routing key"


def test_same_routing_fields_coalesce() -> None:
    base = {
        "session_key": "s", "platform": "discord", "chat_type": "dm",
        "chat_id": "c1", "thread_id": "t1", "user_id": "u1",
    }
    assert batch_key(dict(base)) == batch_key(dict(base))

# ----------------------------------------------- preserved state-machine units
# Moved from test_pending_registry_properties.py, which skips entirely when
# Hypothesis is absent.

def test_deliver_without_claim_is_noop() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        reg.deliver("proc_a")
        assert reg.snapshot()["proc_a"] is State.PENDING

    run_sync(_test())


def test_claim_of_already_claimed_by_other_batch_is_rejected() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        claimed, _skipped = reg.claim_batch(["proc_a"], "b1")
        assert len(claimed) == 1 and claimed[0]["identity"] == "proc_a"
        claimed2, skipped2 = reg.claim_batch(["proc_a"], "b2")
        assert claimed2 == []
        assert skipped2 == ["proc_a"]

    run_sync(_test())


def test_terminal_entry_cannot_be_reclaimed() -> None:
    async def _test():
        reg = Registry()
        reg.enqueue("proc_a", ("r1",), _payload())
        reg.claim_batch(["proc_a"], "b1")
        reg.deliver("proc_a")
        claimed, skipped = reg.claim_batch(["proc_a"], "b2")
        assert claimed == []
        assert skipped == ["proc_a"]

    run_sync(_test())
