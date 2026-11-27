"""A stale interrupt bit on a recycled thread ident must not kill the next tool worker (#135753)."""

import threading
from types import SimpleNamespace

import pytest

from agent.tool_executor import _registered_tool_worker
from tools.interrupt import is_interrupted, set_interrupt


@pytest.fixture
def agent():
    return SimpleNamespace(_tool_worker_threads=set(), _tool_worker_threads_lock=threading.Lock())


def _in_thread(fn):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", fn()))
    t.start()
    t.join(5)
    return out["v"]


def test_stale_bit_on_recycled_ident_is_cleared_on_worker_entry(agent):
    def worker():
        # Simulate interrupt() aimed at an exited thread whose ident this worker now holds.
        set_interrupt(True, threading.get_ident())
        with _registered_tool_worker(agent):
            return is_interrupted()

    assert _in_thread(worker) is False


def test_interrupt_for_registered_worker_still_observed(agent):
    def worker():
        with _registered_tool_worker(agent) as tid:
            assert tid in agent._tool_worker_threads
            set_interrupt(True, tid)  # what AIAgent.interrupt()'s fan-out does
            seen = is_interrupted()
        return seen, is_interrupted()

    assert _in_thread(worker) == (True, False)
