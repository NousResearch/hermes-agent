"""Regression tests for the cooperative runtime primitives in
browser/runtime.py.

Run under plain CPython (no Pyodide): the `js` bridge is optional and pump()
falls back to real sleeps without it. Each test pins one of the wedge/replay
classes hit bringing the real Hermes turn up in-browser:

- non-blocking primitive calls must never SleepBreak a daemon (an incidental
  get_nowait in __del__/finalizers abandoned and replayed whole thread bodies);
- suspension is opt-IN: only loop-named/loop-targeted daemons restart on
  suspend; every other daemon — including unclassified and unnamed ones —
  pump-waits, so no linear body can ever silently replay completed work;
- join() is a completion wait — it pumps rather than suspending the joiner;
- _loop_step tolerates callbacks that re-enter pump() and drain _ready.
"""

import sys
import time
import threading
import queue as _stdlib_queue
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from browser import runtime as br  # noqa: E402


@pytest.fixture
def daemon_ctx():
    """Run the test body as if inside a suspendable daemon thread."""
    prev = getattr(br._in_daemon, "value", False)
    br._in_daemon.value = True
    try:
        yield
    finally:
        br._in_daemon.value = prev


def _drain_sched(duration=0.3):
    """Drive the scheduler for `duration` wall-seconds — suspended daemons
    re-push themselves, so a fixed window is the only bounded drain."""
    end = time.monotonic() + duration
    while time.monotonic() < end:
        fns = br._sched.due()
        for fn in fns:
            fn()
        if not fns:
            nd = br._sched.next_deadline()
            if nd is None:
                break
            time.sleep(min(max(nd - time.monotonic(), 0.001), 0.02))


class TestNonBlockingCalls:
    def test_queue_get_nowait_in_daemon_raises_empty_not_sleepbreak(self, daemon_ctx):
        q = br._make_queue_shim()()
        with pytest.raises(_stdlib_queue.Empty):
            q.get_nowait()

    def test_queue_get_timeout_zero_raises_empty_not_sleepbreak(self, daemon_ctx):
        q = br._make_queue_shim()()
        with pytest.raises(_stdlib_queue.Empty):
            q.get(block=True, timeout=0)

    def test_queue_blocking_get_returns_queued_item_in_daemon(self, daemon_ctx):
        # QueueListener._monitor must see its sentinel: a blocking get with
        # items present returns immediately rather than suspending forever.
        q = br._make_queue_shim()()
        q.put("sentinel")
        assert q.get(block=True) == "sentinel"

    def test_empty_queue_blocking_get_suspends_daemon(self, daemon_ctx):
        q = br._make_queue_shim()()
        with pytest.raises(br.SleepBreak):
            q.get(block=True, timeout=0.5)

    def test_event_wait_timeout_zero_polls_not_suspends(self, daemon_ctx):
        ev = br._EventEmu()
        assert ev.wait(0) is False
        ev.set()
        assert ev.wait(0) is True

    def test_event_wait_unset_suspends_daemon(self, daemon_ctx):
        ev = br._EventEmu()
        with pytest.raises(br.SleepBreak):
            ev.wait(0.5)

    def test_condition_wait_timeout_zero_polls_not_suspends(self, daemon_ctx):
        cond = br._ConditionEmu()
        cond.acquire()
        try:
            assert cond.wait(0) is False
        finally:
            cond.release()

    def test_sleep_zero_does_not_suspend_daemon(self, daemon_ctx):
        br._sleep_pump(0)  # returns; SleepBreak would escape this frame
        br._sleep_pump(0.0)

    def test_positive_sleep_suspends_daemon(self, daemon_ctx):
        with pytest.raises(br.SleepBreak):
            br._sleep_pump(0.05)


class TestThreadSuspensionPolicy:
    def test_suspendable_daemon_restarts_on_suspend(self):
        calls = []
        stop = br._EventEmu()

        def body():
            calls.append(1)
            # Suspendable daemon: Event.wait raises SleepBreak -> _run
            # reschedules -> body restarts from the top (idle-loop shape).
            while not stop.wait(0.01):
                pass

        t = br._ThreadEmu(target=body, name="hermes-test-watcher", daemon=True)
        t.start()
        assert t.is_alive()
        assert len(calls) == 1
        _drain_sched()
        assert len(calls) > 1  # restarted repeatedly
        stop.set()
        _drain_sched()

    def test_linear_daemon_pump_waits_instead_of_suspending(self):
        # prompt-turn-* bodies are linear: a suspend must pump, not replay.
        calls = []

        def body():
            ev = br._EventEmu()
            calls.append(ev.wait(0.01))  # poll-style wait returns False
            calls.append("done")

        t = br._ThreadEmu(target=body, name="prompt-turn-x", daemon=True)
        assert t._suspendable is False
        t.start()  # completes inline; no SleepBreak escapes
        assert calls == [False, "done"]
        assert t._finished.is_set()

    def test_side_agent_threads_are_non_suspendable(self):
        t = br._ThreadEmu(target=lambda: None, name="side-agent-abc", daemon=True)
        assert t._suspendable is False

    def test_unnamed_daemon_is_non_suspendable_by_default(self):
        # The inverted policy's whole point: an unclassified daemon must
        # pump-wait, never SleepBreak-restart — a missed loop daemon parks
        # loudly in start(); a missed linear daemon must never replay.
        t = br._ThreadEmu(target=lambda: None, name="Thread-42", daemon=True)
        assert t._suspendable is False

    def test_unclassified_daemon_body_never_replays(self):
        # A linear daemon body that waits mid-flight completes exactly once.
        calls = []

        def body():
            calls.append("start")
            ev = br._EventEmu()
            ev.wait(0.01)  # pump-waits, no SleepBreak possible
            calls.append("end")

        t = br._ThreadEmu(target=body, name="mystery-daemon", daemon=True)
        t.start()
        assert calls == ["start", "end"]  # ran once, never restarted

    def test_loop_named_threads_stay_suspendable(self):
        t = br._ThreadEmu(target=lambda: None, name="tui-notif-poller-x", daemon=True)
        assert t._suspendable is True

    def test_loop_hinted_targets_are_suspendable(self):
        # Every suspender observed in the live runtime trace must classify.
        class _Targets:
            def _loop(self): pass
            def _monitor(self): pass
            def _worker_loop(self): pass
            def _cleanup_thread_worker(self): pass
            def serve_forever(self): pass
        class PeriodicScheduler:
            def _run(self): pass
        cases = [
            _Targets()._loop, _Targets()._monitor, _Targets()._worker_loop,
            _Targets()._cleanup_thread_worker, _Targets().serve_forever,
            PeriodicScheduler()._run,
        ]
        for fn in cases:
            t = br._ThreadEmu(target=fn, daemon=True)
            assert t._suspendable is True, getattr(fn, "__qualname__", fn)

    def test_linear_targets_stay_non_suspendable(self):
        # Observed linear bodies that must never replay mid-flight.
        class _Linear:
            def _worker(self): pass       # deadline run_bounded_sync._worker
            def _runner(self): pass       # hook callback runner
            def _build(self): pass        # agent build thread
            def _reader(self): pass       # context-read workers
        for fn in (_Linear()._worker, _Linear()._runner,
                   _Linear()._build, _Linear()._reader):
            t = br._ThreadEmu(target=fn, daemon=True)
            assert t._suspendable is False, getattr(fn, "__qualname__", fn)


class TestJoinPumps:
    def test_join_returns_when_joined_finishes_via_scheduled_runs(self):
        # Mirrors the turn wedge: turn thread joins a suspendable ticker
        # daemon; the ticker finishes only on a _sched rerun after stop.set().
        stop = br._EventEmu()

        def ticker_body():
            while not stop.wait(0.01):
                pass

        ticker = br._ThreadEmu(target=ticker_body, name="ticker", daemon=True)
        ticker.start()  # suspends immediately, parked in _sched

        def start_joiner():
            ticker.join()
            joined.append(True)

        joined = []
        joiner = threading.Thread(target=start_joiner, daemon=True)
        joiner.start()
        time.sleep(0.05)
        assert not joined  # still waiting — ticker alive, parked
        stop.set()
        joiner.join(timeout=5)
        assert joined and joiner.is_alive() is False
        assert ticker._finished.is_set()

    def test_join_timeout_returns(self):
        ev = br._EventEmu()

        def body():
            ev.wait(60)  # suspends; daemon parked

        t = br._ThreadEmu(target=body, name="parked-watcher", daemon=True)
        t.start()
        t.join(timeout=0.05)  # must return, not suspend the caller
        ev.set()


class TestLoopStepReentrancy:
    def test_ready_drain_survives_nested_pump(self):
        loop = br.get_loop()
        ran = []

        def inner():
            ran.append("inner")

        def reenter():
            loop.call_soon(inner)
            br.pump(0)  # nested pump drains _ready before outer continues
            ran.append("reenter")

        loop.call_soon(reenter)
        br._loop_step()
        _drain_sched()
        assert "reenter" in ran and "inner" in ran


class TestWaitBlocking:
    def test_returns_when_flag_sets(self):
        flag = {"set": False}
        br._sched.push(0.05, lambda: flag.__setitem__("set", True))
        assert br.wait_blocking(lambda: flag["set"], 2.0) is True

    def test_times_out(self):
        t0 = time.monotonic()
        assert br.wait_blocking(lambda: False, 0.05) is False
        assert time.monotonic() - t0 >= 0.05
