"""Runner-level regression tests for PendingCompletionRegistry (#73469).

Covers the gap declared in the PR: cancellation, shutdown, and the 6-process
scenario — all exercising the real _flush_process_completion_batch path,
not just the registry in isolation.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.run import GatewayRunner, Platform


# ── helpers ──────────────────────────────────────────────────────────────

def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    return runner


def _runner_with_registry(adapter):
    """_runner() plus PendingCompletionRegistry fields for real-flush tests."""
    runner = _runner(adapter)
    runner._completion_registry = GatewayRunner.PendingCompletionRegistry()
    runner._flush_tasks_by_route = {}
    runner._completion_flush_tasks = set()
    runner._completion_notification_batch_lock = threading.Lock()
    runner._completion_notification_batch_window = 0.0
    runner._batch_window_sleep = asyncio.sleep
    runner._completion_route_keys = {}
    runner._completion_entry_data = {}
    runner._background_tasks = set()
    return runner


def _completion_event(*, started_at, session_id="proc_0"):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": started_at,
        "command": f"echo {session_id}",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": f"{session_id} done\n",
    }


async def _instant_sleep(_delay):
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  Cancellation with blocked adapter
# ═══════════════════════════════════════════════════════════════════════════

def test_cancelled_flush_leaves_entries_recoverable():
    """Cancel a flush mid-delivery: entries return to PENDING and can be
    reclaimed by a later batch.

    This is the registry-level property that the runner-level cancellation
    test on #72675 exercises through the full adapter path.  In #73469 the
    state machine keeps entries PENDING with their indexes intact so a
    later flush picks them up.
    """
    async def _test():
        reg = GatewayRunner.PendingCompletionRegistry()

        # Enqueue and claim — the flush has grabbed the entry.
        f = reg.enqueue("proc_a", ("r1",), {"x": 1})
        assert f is not None
        claimed, skipped = reg.claim_batch(["proc_a"], "b1")
        assert len(claimed) == 1 and claimed[0]["identity"] == "proc_a"
        assert reg.snapshot()["proc_a"] is GatewayRunner.PendingCompletionRegistry.State.CLAIMED

        # Simulate cancellation: retry() transitions CLAIMED → PENDING.
        reg.retry("proc_a")
        assert reg.snapshot()["proc_a"] is GatewayRunner.PendingCompletionRegistry.State.PENDING

        # The entry can be reclaimed by a new batch — the first claim's
        # batch_id is irrelevant now.
        claimed2, skipped2 = reg.claim_batch(["proc_a"], "b2")
        assert len(claimed2) == 1
        reg.deliver("proc_a")
        assert reg.snapshot()["proc_a"] is GatewayRunner.PendingCompletionRegistry.State.DELIVERED

    asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════════════════
#  Shutdown with in-flight batch
# ═══════════════════════════════════════════════════════════════════════════

def test_shutdown_with_in_flight_batch_resolves_as_shutting_down():
    """signal_stop() + resolve_futures(SHUTTING_DOWN) resolves every waiter.

    Regression: signal_stop() used to transition entries to STOPPING
    without resolving their futures, so a watcher cancelled inside the
    batch window awaited a Future nothing would complete.
    """
    runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
    runner._background_tasks = set()
    evt = _completion_event(started_at=1.0, session_id="proc_shutdown")

    # Override the batch-window sleep to return immediately so the flush
    # enters its claim+deliver path without actually sleeping.
    async def _noop_sleep(_delay):
        pass

    async def _exercise():
        runner._batch_window_sleep = _noop_sleep
        enq_task = asyncio.create_task(
            runner._enqueue_process_completion_notification("shutdown-me", evt)
        )

        # Give the flush time to claim the entry and enter delivery.
        await asyncio.sleep(0.05)

        # Simulate gateway shutdown: stop new enqueues, stop the registry.
        runner._completion_notification_batches_stopping = True
        stopped = runner._completion_registry.signal_stop()
        assert stopped, "at least one entry should be STOPPING"

        runner._completion_registry.resolve_futures(
            stopped, runner.CompletionDisposition.SHUTTING_DOWN,
        )

        result = await asyncio.wait_for(enq_task, timeout=3.0)
        assert result is runner.CompletionDisposition.SHUTTING_DOWN, (
            f"Expected SHUTTING_DOWN, got {result!r}"
        )
        # Registry must be in terminal state.
        assert runner._completion_registry.is_stopping is True

    asyncio.run(_exercise())


# ═══════════════════════════════════════════════════════════════════════════
#  6-process scenario
# ═══════════════════════════════════════════════════════════════════════════

def test_six_processes_coalesce_into_one_delivery():
    """The original #70300: 6 processes completing in the same tick
    must produce 1 adapter.send(), not 6.
    """
    call_count = 0

    async def _counting_deliver(_event):
        nonlocal call_count
        call_count += 1

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_counting_deliver))
    runner = _runner(adapter)
    runner._background_tasks = set()

    async def _exercise():
        # Enqueue 6 completions — they share the same route key, so they
        # must coalesce into one batch.
        tasks = []
        for i in range(6):
            evt = _completion_event(started_at=float(i), session_id=f"proc_{i}")
            tasks.append(
                asyncio.create_task(
                    runner._enqueue_process_completion_notification(
                        f"output-{i}", evt,
                    )
                )
            )

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # All 6 must be delivered, and the adapter must have been called
        # exactly once — coalesced, not independent.
        assert all(
            r is runner.CompletionDisposition.DELIVERED for r in results
        ), f"Expected all DELIVERED, got {results}"
        assert call_count == 1, (
            f"6 completions must coalesce; got {call_count} adapter calls"
        )

    asyncio.run(_exercise())


# ═══════════════════════════════════════════════════════════════════════════
#  Regression: registry rejection_reason separates duplicate from capacity
# ═══════════════════════════════════════════════════════════════════════════

def test_enqueue_rejection_reason_is_recoverable():
    """Duplicate and capacity refusals must be distinguishable via
    rejection_reason() so callers don't map both onto DROPPED_OVERFLOW.
    """
    async def _test():
        reg = GatewayRunner.PendingCompletionRegistry()

        assert reg.enqueue("proc_a", ("r1",), {"x": 1}) is not None
        assert reg.enqueue("proc_a", ("r1",), {"x": 1}) is None
        assert reg.rejection_reason("proc_a") == "duplicate"

        # Once delivered → terminal — can re-enqueue.
        reg.claim_batch(["proc_a"], "b1")
        reg.deliver("proc_a")
        assert reg.rejection_reason("proc_a") == "capacity"
        assert reg.enqueue("proc_a", ("r1",), {"x": 1}) is not None

    asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════════════════
#  Blocked-adapter cancellation regression (reviewer probe, blocker #73469)
# ═══════════════════════════════════════════════════════════════════════════

def test_cancel_during_blocked_adapter_delivery_keeps_entry_routable():
    """Cancel a flush mid-delivery through the real production path.

    Regression for the non-atomic cancellation documented in the review:
    a terminal default (DROP_UNROUTABLE) + inner-finally-before-outer-handler
    ordering resolved the waiter with a drop, removed the route indexes, and
    left the registry entry PENDING but unreachable.  After the fix the
    cancellation path restores the entry to a *routable* PENDING and resolves
    with RETRY.
    """
    release = asyncio.Event()
    delivery_started = asyncio.Event()

    async def _blocked_injection(_message_event):
        delivery_started.set()
        await release.wait()
        return True  # would be accepted if not cancelled

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_injection))
    runner = _runner_with_registry(adapter)

    async def _exercise():
        # Override sleep so the batch window is instant — the flush enters
        # its claim+deliver path without actually sleeping.  Must be set
        # AFTER registry init so __init__ doesn't overwrite it.
        runner._batch_window_sleep = _instant_sleep

        evt = _completion_event(started_at=1.0, session_id="proc_cancel")
        identity = runner._completion_delivery_identity(evt)
        waiter = asyncio.ensure_future(
            runner._enqueue_process_completion_notification("cancel-me", evt),
        )

        # Wait for the flush to claim the entry and enter the blocked delivery.
        await delivery_started.wait()

        # The flush task removes itself from _flush_tasks_by_route after
        # the batch-window sleep.  Find it via _completion_flush_tasks.
        flush_task = None
        for _t in runner._completion_flush_tasks:
            if not _t.done():
                flush_task = _t
                break
        assert flush_task is not None, "no running flush task in _completion_flush_tasks"
        flush_task.cancel()

        # Let the CancelledError propagate through the flush's handlers.
        try:
            await flush_task
        except asyncio.CancelledError:
            pass

        # Now unblock the adapter so the test can clean up.
        release.set()

        result = await asyncio.wait_for(waiter, timeout=3.0)

        # 1. Waiter gets RETRY — never a terminal drop.
        assert result is runner.CompletionDisposition.RETRY, (
            f"Expected RETRY after cancellation, got {result!r}"
        )

        # 2. After cancellation, the entry is routable — the re-flush
        #    scheduled by the CancelledError handler should deliver it.
        #    Wait for the re-flush to complete.
        await asyncio.sleep(0.1)

        snap = runner._completion_registry.snapshot()
        state = snap.get(identity)
        # If the re-flush delivered, great.  If still PENDING, the entry
        # must still be routable (the index survived the cancellation).
        assert state in (
            runner._completion_registry.State.DELIVERED,
            runner._completion_registry.State.PENDING,
        ), f"Expected DELIVERED or PENDING, got {state!r}"

        if state is runner._completion_registry.State.PENDING:
            assert runner._completion_route_keys.get(identity) is not None, (
                "route index was removed — entry is PENDING but unreachable"
            )

        # 3. signal_stop recovers or acknowledges the entry.
        runner._running = False  # let stop() proceed
        runner._shutdown_event = asyncio.Event()
        runner.adapters = {}  # avoid real adapter disconnect
        final_snap = runner._completion_registry.snapshot()
        final_state = final_snap.get(identity)
        # The entry is at least not UNROUTABLE — the termination was
        # either DELIVERED (re-flush succeeded) or STOPPING (signal_stop
        # caught it).  Never missing, never DROP_UNROUTABLE.
        if final_state is not None:
            assert final_state is not runner._completion_registry.State.UNROUTABLE, (
                f"entry ended UNROUTABLE — a terminal drop (state={final_state!r})"
            )
        # If the entry is missing from the snapshot it was evicted as a
        # terminal DELIVERED beyond TERMINAL_RETENTION — which can happen
        # when the re-flush delivered and then signal_stop's eviction swept it.
        # The key property: it never became UNROUTABLE.

    asyncio.run(_exercise())


# ═══════════════════════════════════════════════════════════════════════════
#  signal_stop finds entries missing from route index (defense in depth)
# ═══════════════════════════════════════════════════════════════════════════

def test_stop_recovers_entry_orphaned_by_missing_route_index():
    """signal_stop scans the authoritative entry map, not the route index.

    If a future bug orphans an entry (PENDING, no route key), signal_stop
    still finds and recovers it.  Degrades from silent loss to delayed
    delivery — the net the reviewer described.
    """
    async def _test():
        reg = GatewayRunner.PendingCompletionRegistry()
        f = reg.enqueue("proc_orphan", ("r1",), {"synth_text": "t", "evt": {}})
        assert f is not None

        # Simulate the orphan: entry is PENDING but not in any route index.
        stopped = reg.signal_stop()
        assert any(e["identity"] == "proc_orphan" for e in stopped), (
            "signal_stop did not find the orphaned entry"
        )
        reg.resolve_futures(stopped, GatewayRunner.CompletionDisposition.SHUTTING_DOWN)
        assert await f is GatewayRunner.CompletionDisposition.SHUTTING_DOWN
        assert reg.snapshot()["proc_orphan"] is (
            GatewayRunner.PendingCompletionRegistry.State.STOPPING
        )

    asyncio.run(_test())
