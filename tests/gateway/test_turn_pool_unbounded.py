"""An accepted turn must never queue behind other turn bodies on the gateway's turn pool.

A turn body holds its executor thread for the whole turn (every tool call blocks). With a finite
pool (it was 10 threads), a burst of turns after a restart (resumed sessions plus new messages)
filled it, and every later turn, including the user's next message, waited in the executor queue
for minutes with no log line naming the wait. Concurrency is bounded at admission, not here.
"""

from __future__ import annotations

import threading

import pytest

from gateway.run import GatewayRunner


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._executor_lock = threading.Lock()
    runner._executor = None
    runner._housekeeping_executor = None
    runner._executor_closing = False
    return runner


def test_a_new_turn_starts_while_many_are_parked():
    runner = _runner()
    release = threading.Event()
    parked = threading.Semaphore(0)

    def parked_body():
        parked.release()
        assert release.wait(30)

    ex = runner._get_executor()
    try:
        for _ in range(40):
            ex.submit(parked_body)
        for _ in range(40):
            assert parked.acquire(timeout=10), "a parked body never started"
        started = threading.Event()
        ex.submit(started.set)
        assert started.wait(2), "the 41st turn body queued behind 40 parked ones"
    finally:
        release.set()
        runner._shutdown_executor(drain_timeout=5)


def test_shutdown_counts_a_live_turn_worker(monkeypatch):
    runner = _runner()
    wedge = threading.Event()
    entered = threading.Event()

    def wedged():
        entered.set()
        assert wedge.wait(30)

    ex = runner._get_executor()
    ex.submit(wedged)
    assert entered.wait(5)

    # At the OS thread limit Thread.start raises; that submit must fail without leaving an
    # unstarted thread behind for shutdown to join.
    def refuse(self):
        raise RuntimeError("can't start new thread")

    with monkeypatch.context() as m:
        m.setattr(threading.Thread, "start", refuse)
        with pytest.raises(RuntimeError):
            ex.submit(lambda: None)
    try:
        live = runner._shutdown_executor(drain_timeout=0.2)
    finally:
        wedge.set()
    assert live == 1, f"wedged turn worker not reported as live (got {live})"
